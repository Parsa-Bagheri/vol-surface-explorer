# UI/UX Improvement Ideas

Generated: 2026-05-26

## Already Applied In This Pass

- Replaced paired min/max sliders with one dual-ended range control for strike and one for DTE.
- Made the range endpoints directly editable so users can type exact numbers.
- Moved the page title to a centered, bold top position and removed the descriptive subtitle.
- Reduced the chart title clutter by keeping the title in the page chrome and removing the long Plotly title.
- Put only the essential metrics inside the chart shell: spot, requested DTE, and available DTE.
- Moved secondary metrics and diagnostics into a single collapsible `Advanced info` panel below the graph.
- Restyled the page toward a professional brutalist feel with washed blacks, greys, sharper borders, and a navy chart shell.
- Changed the plot area to a desktop 4:3 shape so the surface has more vertical room.

## Additional Improvements Worth Considering

### Save Named Presets

Add small preset chips for common views such as `Near-term`, `Earnings window`, and `Quarterly`. This would reduce repeated slider work without hiding the manual controls.

### Add A Loading State

Surface builds can take time because option chains are fetched live. A strong loading state inside the navy chart area would make the app feel responsive and prevent users from wondering whether the button worked.

### Add Inline Data Recency

Show the latest option trade timestamp and fetch timestamp in `Advanced info`. This would make data freshness easier to judge without adding clutter to the primary view.

### Make Smoothing Sticky

Persist the smoothing preference in local storage. Users comparing tickers would not need to reset the same display preference each time.

### Add Keyboard-Friendly Range Editing

The editable endpoints already allow typing. Adding small focus states and arrow-key increments with clear units would make precise changes faster for power users.

### Add A Compact Row-Level Export

A CSV export of the exact surface nodes used in the plot would help users audit or reproduce a surface in notebooks without scraping diagnostics.

### Add View Controls For The 3D Camera

Buttons for `Front`, `Top`, and `Reset` would make the 3D surface easier to inspect. These should live inside the chart shell as small icon buttons, not as another row of text controls.

### Add A Sparse-Data Empty State

When a ticker/range has too few eligible rows, show which condition failed most often: no expirations, no strikes in range, stale quotes, or unreliable IV. This would make failures easier to fix.

### Add A Surface Smoothness Toggle Near The Chart

The smoothing checkbox currently belongs with request controls. If users often compare raw nodes to the smoothed surface, a small chart-level toggle could make that comparison faster after the first build.
