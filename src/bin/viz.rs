/// gpusim live visualizer — attach to any running simulation at any time.
///
/// Run in a separate terminal:
///   cargo run --bin viz
///
/// Polls /tmp/gpusim_live.json every 200ms and renders a live TUI dashboard:
///
///   Single-GPU mode:
///     ┌ header: kernel / policy / status ──────────────────────────┐
///     │ SM heatmap (one cell per SM)  │ Stats: occupancy, blocks … │
///     │ q/esc: quit  …footer…                                      │
///
///   Cluster mode (cluster_mode = true in the snapshot):
///     ┌ header: kernel / policy / status / active device ──────────┐
///     │ SM heatmap (active GPU)       │ Stats: occupancy, blocks … │
///     │ Cluster topology: node × GPU grid, last transfer, collective│
///     │ q/esc: quit  …footer…                                      │
///
/// Press q or Esc to quit. The simulation keeps running unaffected.
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use gpusim::metrics::{read_metrics, LiveMetrics};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
    Frame, Terminal,
};
use std::{io, time::Duration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run(&mut terminal);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn run(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let metrics = read_metrics();
        terminal.draw(|f| render(f, metrics.as_ref()))?;

        // Non-blocking: poll for 200ms, then redraw regardless
        if event::poll(Duration::from_millis(200))? {
            if let Event::Key(key) = event::read()? {
                if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) {
                    break;
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Top-level layout
// ---------------------------------------------------------------------------

fn render(f: &mut Frame, metrics: Option<&LiveMetrics>) {
    let area = f.area();
    let is_cluster = metrics.map(|m| m.cluster_mode).unwrap_or(false);

    // Cluster panel height: 2 borders + 1 topology header + num_nodes rows +
    // 1 blank + up to 2 event lines. Minimum 7, capped at 14.
    let cluster_height = if is_cluster {
        let num_nodes = metrics.map(|m| m.num_nodes).unwrap_or(2);
        (num_nodes as u16 + 6).clamp(7, 14)
    } else {
        0
    };

    let rows = if is_cluster {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),              // header
                Constraint::Min(8),                 // heatmap + stats
                Constraint::Length(cluster_height), // cluster panel
                Constraint::Length(1),              // footer
            ])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // header
                Constraint::Min(0),    // heatmap + stats
                Constraint::Length(1), // footer
            ])
            .split(area)
    };

    render_header(f, rows[0], metrics);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(rows[1]);

    render_heatmap(f, cols[0], metrics);
    render_stats(f, cols[1], metrics);

    if is_cluster {
        render_cluster(f, rows[2], metrics.unwrap());
        render_footer(f, rows[3]);
    } else {
        render_footer(f, rows[2]);
    }
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

fn render_header(f: &mut Frame, area: Rect, metrics: Option<&LiveMetrics>) {
    let block = Block::default()
        .title(Span::styled(
            " ⚡ gpusim live monitor ",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let (name, policy, status, device) = metrics
        .map(|m| {
            let dev = if m.cluster_mode && !m.active_device.is_empty() {
                m.active_device.as_str()
            } else {
                ""
            };
            (m.kernel_name.as_str(), m.scheduling_policy.as_str(), m.status.as_str(), dev)
        })
        .unwrap_or(("—", "—", "idle", ""));

    let status_color = match status {
        "running" => Color::Green,
        "complete" => Color::Cyan,
        "transfer" => Color::Magenta,
        "collective" => Color::Blue,
        _ => Color::DarkGray,
    };

    let mut spans = vec![
        Span::styled("  kernel: ", Style::default().fg(Color::DarkGray)),
        Span::styled(name, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled("   policy: ", Style::default().fg(Color::DarkGray)),
        Span::styled(policy, Style::default().fg(Color::Cyan)),
        Span::styled("   status: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            status.to_uppercase(),
            Style::default().fg(status_color).add_modifier(Modifier::BOLD),
        ),
    ];

    if !device.is_empty() {
        spans.push(Span::styled("   device: ", Style::default().fg(Color::DarkGray)));
        spans.push(Span::styled(
            device,
            Style::default().fg(Color::Yellow),
        ));
    }

    f.render_widget(Paragraph::new(Line::from(spans)), inner);
}

// ---------------------------------------------------------------------------
// SM heatmap
// ---------------------------------------------------------------------------

fn render_heatmap(f: &mut Frame, area: Rect, metrics: Option<&LiveMetrics>) {
    // When in cluster mode, label the panel with the active GPU
    let title = metrics
        .filter(|m| m.cluster_mode && !m.active_device.is_empty())
        .map(|m| format!(" SM Utilization ({}) ", m.active_device))
        .unwrap_or_else(|| " SM Utilization ".to_string());

    let block = Block::default().title(title).borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let sm_active: Vec<u32> = metrics
        .map(|m| m.sm_active_blocks.clone())
        .unwrap_or_else(|| vec![0u32; 132]);

    // Fit as many SMs per row as the panel width allows (each SM = 2 chars + 1 space)
    let sms_per_row = ((inner.width as usize).saturating_sub(1) / 3).max(1);

    // Legend line at top
    let legend = Line::from(vec![
        Span::styled("██", Style::default().fg(Color::Green)),
        Span::raw(" active   "),
        Span::styled("░░", Style::default().fg(Color::DarkGray)),
        Span::raw(" idle"),
    ]);

    let mut lines: Vec<Line> = vec![legend, Line::raw("")];

    for row in sm_active.chunks(sms_per_row) {
        let spans: Vec<Span> = row
            .iter()
            .flat_map(|&active| {
                let (symbol, color) =
                    if active > 0 { ("██", Color::Green) } else { ("░░", Color::DarkGray) };
                vec![Span::styled(symbol, Style::default().fg(color)), Span::raw(" ")]
            })
            .collect();
        lines.push(Line::from(spans));
    }

    // Show SM count summary below the grid
    let active_count = sm_active.iter().filter(|&&b| b > 0).count();
    lines.push(Line::raw(""));
    lines.push(Line::from(vec![Span::styled(
        format!("  {}/{} SMs active", active_count, sm_active.len()),
        Style::default().fg(Color::DarkGray),
    )]));

    f.render_widget(Paragraph::new(lines), inner);
}

// ---------------------------------------------------------------------------
// Stats panel
// ---------------------------------------------------------------------------

fn render_stats(f: &mut Frame, area: Rect, metrics: Option<&LiveMetrics>) {
    let block = Block::default().title(" Stats ").borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // occupancy gauge
            Constraint::Length(1), // spacer
            Constraint::Length(2), // blocks gauge
            Constraint::Length(1), // spacer
            Constraint::Min(0),    // text stats
        ])
        .split(inner);

    match metrics {
        None => {
            let msg = Paragraph::new(vec![
                Line::raw(""),
                Line::from(Span::styled(
                    "  No simulation running.",
                    Style::default().fg(Color::DarkGray),
                )),
                Line::from(Span::styled(
                    "  Start gpusim to see live data.",
                    Style::default().fg(Color::DarkGray),
                )),
            ]);
            f.render_widget(msg, inner);
        }
        Some(m) => {
            // Occupancy gauge
            let occ_pct = (m.theoretical_occupancy * 100.0).clamp(0.0, 100.0) as u16;
            let occ_color = match occ_pct {
                0..=33 => Color::Red,
                34..=66 => Color::Yellow,
                _ => Color::Green,
            };
            let occ_gauge = Gauge::default()
                .block(Block::default().title("Occupancy"))
                .gauge_style(Style::default().fg(occ_color))
                .percent(occ_pct)
                .label(format!("{:.1}%", m.theoretical_occupancy * 100.0));
            f.render_widget(occ_gauge, rows[0]);

            // Block progress gauge
            let blk_pct = if m.blocks_total > 0 {
                ((m.blocks_executed as f32 / m.blocks_total as f32) * 100.0) as u16
            } else {
                0
            };
            let blk_gauge = Gauge::default()
                .block(Block::default().title("Blocks"))
                .gauge_style(Style::default().fg(Color::Blue))
                .percent(blk_pct)
                .label(format!("{} / {}", m.blocks_executed, m.blocks_total));
            f.render_widget(blk_gauge, rows[2]);

            // Text stats
            let text = vec![
                Line::from(vec![
                    Span::styled("Warps:      ", Style::default().fg(Color::DarkGray)),
                    Span::raw(m.warps_executed.to_string()),
                ]),
                Line::from(vec![
                    Span::styled("Threads:    ", Style::default().fg(Color::DarkGray)),
                    Span::raw(m.threads_executed.to_string()),
                ]),
                Line::from(vec![
                    Span::styled("Max blk/SM: ", Style::default().fg(Color::DarkGray)),
                    Span::raw(m.max_blocks_per_sm.to_string()),
                ]),
                Line::from(vec![
                    Span::styled("Limiter:    ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        m.occupancy_limiter.clone(),
                        Style::default().fg(Color::Yellow),
                    ),
                ]),
                Line::raw(""),
                Line::from(vec![
                    Span::styled("Grid:   ", Style::default().fg(Color::DarkGray)),
                    Span::raw(format!("({},{},{})", m.grid[0], m.grid[1], m.grid[2])),
                ]),
                Line::from(vec![
                    Span::styled("Block:  ", Style::default().fg(Color::DarkGray)),
                    Span::raw(format!("({},{},{})", m.block[0], m.block[1], m.block[2])),
                ]),
            ];
            f.render_widget(Paragraph::new(text), rows[4]);
        }
    }
}

// ---------------------------------------------------------------------------
// Cluster panel  (only shown when cluster_mode = true)
// ---------------------------------------------------------------------------

fn render_cluster(f: &mut Frame, area: Rect, m: &LiveMetrics) {
    let title = format!(
        " Cluster: {} nodes × {} GPUs ({} total)  \
         NVLink {:.0} GB/s │ InfiniBand {:.0} GB/s ",
        m.num_nodes,
        m.gpus_per_node,
        m.num_nodes * m.gpus_per_node,
        m.nvlink_bw_gb_s,
        m.infiniband_bw_gb_s,
    );
    let block = Block::default().title(title).borders(Borders::ALL);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();

    // ------------------------------------------------------------------
    // Topology grid: one row per node, one cell per GPU
    // Active kernel GPU highlighted in yellow; others in dark gray.
    // ------------------------------------------------------------------
    for node_idx in 0..m.num_nodes {
        let mut spans: Vec<Span> = vec![Span::styled(
            format!("  Node {:2}  ", node_idx),
            Style::default().fg(Color::DarkGray),
        )];

        for gpu_idx in 0..m.gpus_per_node {
            let device_str = format!("node{}:gpu{}", node_idx, gpu_idx);
            let is_active = m.active_device == device_str;

            let (symbol, color, bold) = if is_active {
                // Bright yellow block = GPU that ran / is running the kernel
                ("██", Color::Yellow, true)
            } else {
                ("░░", Color::DarkGray, false)
            };

            let style = if bold {
                Style::default().fg(color).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(color)
            };
            spans.push(Span::styled(symbol, style));
            spans.push(Span::raw(" "));
        }

        // Legend hint on the first node row
        if node_idx == 0 {
            spans.push(Span::styled(
                "  ██=kernel  ░░=idle",
                Style::default().fg(Color::DarkGray),
            ));
        }

        lines.push(Line::from(spans));
    }

    lines.push(Line::raw(""));

    // ------------------------------------------------------------------
    // Last point-to-point transfer
    // ------------------------------------------------------------------
    if let Some(t) = &m.last_transfer {
        let chan_color = match t.channel.as_str() {
            "NVLink" => Color::Green,
            "InfiniBand" => Color::Blue,
            _ => Color::DarkGray,
        };
        lines.push(Line::from(vec![
            Span::styled("  Transfer   ", Style::default().fg(Color::DarkGray)),
            Span::styled(t.src.clone(), Style::default().fg(Color::Cyan)),
            Span::raw(" → "),
            Span::styled(t.dst.clone(), Style::default().fg(Color::Cyan)),
            Span::raw(format!("   {:.1} MB   {:.2} ms   ", t.bytes_mb, t.time_ms)),
            Span::styled(
                format!("{:.1} GB/s", t.bandwidth_gb_s),
                Style::default().fg(Color::Green),
            ),
            Span::styled(format!("  ({})", t.channel), Style::default().fg(chan_color)),
        ]));
    } else {
        lines.push(Line::from(Span::styled(
            "  Transfer   —",
            Style::default().fg(Color::DarkGray),
        )));
    }

    // ------------------------------------------------------------------
    // Last collective operation
    // ------------------------------------------------------------------
    if let Some(c) = &m.last_collective {
        let eff_color = match c.efficiency_pct as u32 {
            0..=60 => Color::Red,
            61..=85 => Color::Yellow,
            _ => Color::Green,
        };
        lines.push(Line::from(vec![
            Span::styled("  Collective ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}/{}", c.operation, c.algorithm),
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(
                "   {} GPUs   {:.1} MB/GPU   {:.2} ms   ",
                c.num_gpus, c.bytes_per_gpu_mb, c.time_ms
            )),
            Span::styled(
                format!("{:.1} GB/s", c.bus_bw_gb_s),
                Style::default().fg(Color::Green),
            ),
            Span::styled(
                format!("   {:.1}%", c.efficiency_pct),
                Style::default().fg(eff_color),
            ),
        ]));
    } else {
        lines.push(Line::from(Span::styled(
            "  Collective —",
            Style::default().fg(Color::DarkGray),
        )));
    }

    f.render_widget(Paragraph::new(lines), inner);
}

// ---------------------------------------------------------------------------
// Footer
// ---------------------------------------------------------------------------

fn render_footer(f: &mut Frame, area: Rect) {
    let text = Paragraph::new(Span::styled(
        "  q / esc: quit    auto-refreshes every 200ms    reads /tmp/gpusim_live.json",
        Style::default().fg(Color::DarkGray),
    ));
    f.render_widget(text, area);
}
