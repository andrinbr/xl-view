//! Read-only Wayland color-management output discovery.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use wayland_client::protocol::{wl_output, wl_registry};
use wayland_client::{Connection, Dispatch, Proxy, QueueHandle};
use wayland_protocols::wp::color_management::v1::client::{
    wp_color_management_output_v1, wp_color_manager_v1, wp_image_description_info_v1,
    wp_image_description_v1,
};

use super::{HdrTarget, HdrUnavailableReason, wayland_target};
use crate::gpu::DisplayChangeNotifier;

// Retain the raw protocol input so diagnostics and target selection share one
// source of truth without re-querying the compositor.
#[derive(Clone, Debug, PartialEq)]
struct OutputSnapshot {
    name: Option<String>,
    luminances: Option<WaylandLuminances>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WaylandLuminances {
    // The protocol transports minimum luminance as cd/m² multiplied by 10,000;
    // maximum and reference white are unscaled integer cd/m² values.
    minimum_nits_times_10000: u32,
    maximum_nits: u32,
    reference_white_nits: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Snapshot {
    color_management_present: bool,
    outputs: BTreeMap<u32, OutputSnapshot>,
}

struct WaylandDisplayInformation<'a> {
    monitor_name: Option<&'a str>,
    color_management_present: bool,
    outputs: &'a BTreeMap<u32, OutputSnapshot>,
}

// A manual implementation avoids suppressing dead-code warnings for fields
// whose only consumer is diagnostic Debug formatting.
impl fmt::Debug for WaylandDisplayInformation<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WaylandDisplayInformation")
            .field("monitor_name", &self.monitor_name)
            .field("color_management_present", &self.color_management_present)
            .field("outputs", &self.outputs)
            .finish()
    }
}

impl Snapshot {
    fn target(&self, monitor_name: Option<&str>) -> HdrTarget {
        if !self.color_management_present {
            return Err(HdrUnavailableReason::MissingColorManagement);
        }
        if let Some(monitor_name) = monitor_name
            && let Some(output) = self
                .outputs
                .values()
                .find(|output| output.name.as_deref() == Some(monitor_name))
        {
            return target_from_luminances(output.luminances);
        }
        if self.outputs.len() == 1 {
            return target_from_luminances(self.outputs.values().next().unwrap().luminances);
        }
        Err(HdrUnavailableReason::OutputUnmatched)
    }
}

pub(in crate::gpu) struct DisplayTargetProvider {
    snapshot: Arc<Mutex<Snapshot>>,
}

impl DisplayTargetProvider {
    // Rendering also refreshes winit's current-monitor association.
    pub(in crate::gpu) const QUERY_BEFORE_RENDER: bool = true;

    pub(in crate::gpu) fn new(notify_changed: DisplayChangeNotifier) -> Self {
        let snapshot = Arc::new(Mutex::new(Snapshot::default()));
        let provider = Self {
            snapshot: Arc::clone(&snapshot),
        };
        let connection = match Connection::connect_to_env() {
            Ok(connection) => connection,
            Err(error) => {
                // Winit consumes an inherited WAYLAND_SOCKET while creating its
                // event loop, so HDR discovery requires a separately connectable
                // WAYLAND_DISPLAY endpoint.
                tracing::warn!(%error, "cannot connect to Wayland for color-management queries");
                return provider;
            }
        };
        let mut event_queue = connection.new_event_queue();
        let queue = event_queue.handle();
        let _registry = connection.display().get_registry(&queue, ());
        let mut state = State {
            manager: None,
            outputs: BTreeMap::new(),
            snapshot,
            notify_changed,
        };

        // Four roundtrips cap synchronous bootstrap latency; a slower
        // compositor may publish an incomplete first snapshot, then the event
        // thread completes it and emits the normal changed notification.
        for _ in 0..4 {
            if let Err(error) = event_queue.roundtrip(&mut state) {
                tracing::warn!(%error, "cannot query Wayland color-management output information");
                return provider;
            }
        }
        if let Err(error) = std::thread::Builder::new()
            .name("xl-view-wayland-color-management".to_owned())
            .spawn(
                move || {
                    while event_queue.blocking_dispatch(&mut state).is_ok() {}
                },
            )
        {
            tracing::warn!(%error, "cannot monitor Wayland color-management output changes");
        }
        provider
    }

    pub(in crate::gpu) fn target(&self, monitor_name: Option<&str>) -> HdrTarget {
        self.snapshot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .target(monitor_name)
    }

    pub(in crate::gpu) fn raw_display_information(&self, monitor_name: Option<&str>) -> String {
        let snapshot = self
            .snapshot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Include every output, not just the current match. Names and raw
        // luminances make an OutputUnmatched result diagnosable.
        format!(
            "{:#?}",
            WaylandDisplayInformation {
                monitor_name,
                color_management_present: snapshot.color_management_present,
                outputs: &snapshot.outputs,
            }
        )
    }
}

struct OutputState {
    output: wl_output::WlOutput,
    color_output: Option<wp_color_management_output_v1::WpColorManagementOutputV1>,
    name: Option<String>,
    // Last complete description published to readers.
    luminances: Option<WaylandLuminances>,
    generation: u64,
    // Description information arrives across several events and becomes
    // authoritative only when the compositor sends Done.
    pending_luminances: Option<WaylandLuminances>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutputKey(u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DescriptionKey {
    output: u32,
    generation: u64,
}

fn target_from_luminances(luminances: Option<WaylandLuminances>) -> HdrTarget {
    luminances.map_or_else(
        || Err(HdrUnavailableReason::MissingDisplayInformation),
        |luminances| {
            wayland_target(
                Some(luminances.reference_white_nits),
                Some(luminances.maximum_nits),
            )
        },
    )
}

struct State {
    manager: Option<wp_color_manager_v1::WpColorManagerV1>,
    outputs: BTreeMap<u32, OutputState>,
    snapshot: Arc<Mutex<Snapshot>>,
    notify_changed: DisplayChangeNotifier,
}

impl State {
    fn add_color_output(&mut self, global_name: u32, queue: &QueueHandle<Self>) {
        let Some(manager) = self.manager.as_ref() else {
            return;
        };
        let Some(output) = self.outputs.get_mut(&global_name) else {
            return;
        };
        if output.color_output.is_some() {
            return;
        }
        let color_output = manager.get_output(&output.output, queue, OutputKey(global_name));
        output.color_output = Some(color_output);
        self.refresh_description(global_name, queue);
    }

    fn refresh_description(&mut self, global_name: u32, queue: &QueueHandle<Self>) {
        let Some(output) = self.outputs.get_mut(&global_name) else {
            return;
        };
        let Some(color_output) = output.color_output.as_ref() else {
            return;
        };
        output.generation = output.generation.wrapping_add(1);
        output.pending_luminances = None;
        let key = DescriptionKey {
            output: global_name,
            generation: output.generation,
        };
        color_output.get_image_description(queue, key);
    }

    fn publish(&self) {
        let next = Snapshot {
            color_management_present: self.manager.is_some(),
            outputs: self
                .outputs
                .iter()
                .map(|(&name, output)| {
                    (
                        name,
                        OutputSnapshot {
                            name: output.name.clone(),
                            luminances: output.luminances,
                        },
                    )
                })
                .collect(),
        };
        let changed = {
            let mut current = self
                .snapshot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if *current == next {
                false
            } else {
                *current = next;
                true
            }
        };
        if changed {
            (self.notify_changed)();
        }
    }

    fn commit_luminances(&mut self, key: DescriptionKey, luminances: Option<WaylandLuminances>) {
        let Some(output) = self.outputs.get_mut(&key.output) else {
            return;
        };
        if output.generation != key.generation {
            return;
        }
        output.luminances = luminances;
        self.publish();
    }

    fn current_output_mut(&mut self, key: DescriptionKey) -> Option<&mut OutputState> {
        self.outputs
            .get_mut(&key.output)
            .filter(|output| output.generation == key.generation)
    }
}

impl Dispatch<wl_registry::WlRegistry, ()> for State {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        (): &(),
        _: &Connection,
        queue: &QueueHandle<Self>,
    ) {
        match event {
            wl_registry::Event::Global {
                name,
                interface,
                version,
            } if interface == wp_color_manager_v1::WpColorManagerV1::interface().name => {
                state.manager = Some(registry.bind(name, version.min(3), queue, ()));
                let outputs = state.outputs.keys().copied().collect::<Vec<_>>();
                for output in outputs {
                    state.add_color_output(output, queue);
                }
                state.publish();
            }
            wl_registry::Event::Global {
                name,
                interface,
                version,
            } if interface == wl_output::WlOutput::interface().name => {
                let output = registry.bind(name, version.min(4), queue, OutputKey(name));
                state.outputs.insert(
                    name,
                    OutputState {
                        output,
                        color_output: None,
                        name: None,
                        luminances: None,
                        generation: 0,
                        pending_luminances: None,
                    },
                );
                state.add_color_output(name, queue);
                state.publish();
            }
            wl_registry::Event::GlobalRemove { name } => {
                state.outputs.remove(&name);
                state.publish();
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_output::WlOutput, OutputKey> for State {
    fn event(
        state: &mut Self,
        _: &wl_output::WlOutput,
        event: wl_output::Event,
        key: &OutputKey,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_output::Event::Name { name } = event
            && let Some(output) = state.outputs.get_mut(&key.0)
            && output.name.as_deref() != Some(&name)
        {
            output.name = Some(name);
            state.publish();
        }
    }
}

wayland_client::delegate_noop!(State: ignore wp_color_manager_v1::WpColorManagerV1);

impl Dispatch<wp_color_management_output_v1::WpColorManagementOutputV1, OutputKey> for State {
    fn event(
        state: &mut Self,
        _: &wp_color_management_output_v1::WpColorManagementOutputV1,
        event: wp_color_management_output_v1::Event,
        key: &OutputKey,
        _: &Connection,
        queue: &QueueHandle<Self>,
    ) {
        if let wp_color_management_output_v1::Event::ImageDescriptionChanged = event {
            state.refresh_description(key.0, queue);
        }
    }
}

impl Dispatch<wp_image_description_v1::WpImageDescriptionV1, DescriptionKey> for State {
    fn event(
        state: &mut Self,
        description: &wp_image_description_v1::WpImageDescriptionV1,
        event: wp_image_description_v1::Event,
        key: &DescriptionKey,
        _: &Connection,
        queue: &QueueHandle<Self>,
    ) {
        match event {
            wp_image_description_v1::Event::Ready { .. }
            | wp_image_description_v1::Event::Ready2 { .. } => {
                if let Some(output) = state.current_output_mut(*key) {
                    output.pending_luminances = None;
                    description.get_information(queue, *key);
                    // Requests are ordered, and the new information object
                    // remains valid after the description is destroyed.
                }
                description.destroy();
            }
            wp_image_description_v1::Event::Failed { .. } => {
                state.commit_luminances(*key, None);
                description.destroy();
            }
            _ => {}
        }
    }
}

impl Dispatch<wp_image_description_info_v1::WpImageDescriptionInfoV1, DescriptionKey> for State {
    fn event(
        state: &mut Self,
        _: &wp_image_description_info_v1::WpImageDescriptionInfoV1,
        event: wp_image_description_info_v1::Event,
        key: &DescriptionKey,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wp_image_description_info_v1::Event::Luminances {
                min_lum,
                max_lum,
                reference_lum,
            } => {
                if let Some(output) = state.current_output_mut(*key) {
                    output.pending_luminances = Some(WaylandLuminances {
                        minimum_nits_times_10000: min_lum,
                        maximum_nits: max_lum,
                        reference_white_nits: reference_lum,
                    });
                }
            }
            wp_image_description_info_v1::Event::Done => {
                if let Some(output) = state.current_output_mut(*key) {
                    // Do not expose a partial information sequence. Done is the
                    // protocol boundary at which all description events apply.
                    let luminances = output.pending_luminances.take();
                    state.commit_luminances(*key, luminances);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn luminances(reference_white_nits: u32, maximum_nits: u32) -> WaylandLuminances {
        WaylandLuminances {
            minimum_nits_times_10000: 1,
            maximum_nits,
            reference_white_nits,
        }
    }

    #[test]
    fn output_matching_uses_exact_name_then_single_output_rule() {
        let single = Snapshot {
            color_management_present: true,
            outputs: [(
                7,
                OutputSnapshot {
                    name: None,
                    luminances: Some(luminances(200, 600)),
                },
            )]
            .into(),
        };
        assert_eq!(single.target(Some("unmatched")), Ok(600.0));

        let multiple = Snapshot {
            color_management_present: true,
            outputs: [
                (
                    7,
                    OutputSnapshot {
                        name: Some("DP-1".to_owned()),
                        luminances: Some(luminances(200, 600)),
                    },
                ),
                (
                    8,
                    OutputSnapshot {
                        name: Some("HDMI-A-1".to_owned()),
                        luminances: Some(luminances(200, 1_000)),
                    },
                ),
            ]
            .into(),
        };
        assert_eq!(multiple.target(Some("HDMI-A-1")), Ok(1_000.0));
        assert_eq!(
            multiple.target(Some("unmatched")).unwrap_err(),
            HdrUnavailableReason::OutputUnmatched
        );
        assert_eq!(
            multiple.target(None).unwrap_err(),
            HdrUnavailableReason::OutputUnmatched
        );
    }

    #[test]
    fn absent_color_management_is_distinct_from_output_matching() {
        assert_eq!(
            Snapshot::default().target(None).unwrap_err(),
            HdrUnavailableReason::MissingColorManagement
        );
    }

    #[test]
    fn protocol_information_assembles_only_complete_valid_integer_luminance() {
        assert_eq!(
            target_from_luminances(None).unwrap_err(),
            HdrUnavailableReason::MissingDisplayInformation
        );
        assert_eq!(
            target_from_luminances(Some(luminances(300, 300))).unwrap_err(),
            HdrUnavailableReason::DisplayReportsSdr
        );
        assert_eq!(
            target_from_luminances(Some(luminances(200, 600))),
            Ok(600.0)
        );
    }
}
