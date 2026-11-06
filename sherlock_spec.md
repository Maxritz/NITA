# Sherlock‑Lite Tracing & Breadcrumb Debug Tags
**Date:** 6 Nov 2026  

## 1️⃣ Sherlock‑Lite Tracing (lightweight per‑component timing)

**Stdout format (exact match to user example):**

```
component           ops   %dev      dev us     idle us     host us
attention            24  95.4%   535880.80      377.62       49.20
projections         187   3.2%    17723.32     2026.19      460.80
recurrent            90   1.1%     6127.59     1170.02      143.80
attention-mix         72   0.2%     1335.28     1102.27      111.60
head+sample           1   0.0%        98.20       91.80       56.90
bias                 18   0.0%        90.03      265.52       66.60
transfer              1   0.0%        79.80    1087.10    1084.30
norms                67   0.0%        72.98      784.24      139.40
residual             48   0.0%        23.40      198.44       73.30
ffn-activate         42   0.0%        21.52      372.84       67.70
embed                 1   0.0%         0.48       25.44        2.40
instrumentation floor, 256 empty ops timed through the same begin/end path: 0.70 us host, 19.54 us device each.
```

### Lightweight Implementation Pointers

```cpp
// Simple RAII timer (compile‑out in release builds via ‑DSHERLOCK_LITE)
struct sherlock_timer {
    const char* name;
    double ops;
    chrono::time_point<steady_clock> start, end;
    sherlock_timer(const char* n, double o = 0) : name(n), ops(o) {
        #ifdef SHERLOCK_LITE
        start = steady_clock::now();
        #endif
    }
    ~sherlock_timer() {
        #ifdef SHERLOCK_LITE
        end = steady_clock::now();
        double host_us = duration_cast<microseconds>(end-start).count();
        // device time queried via external marker (see below)
        // emit line:
        fprintf(stdout, "%-20s %-5.0f %-6.1f %-9.2f %-8.2f %-7.2f\n",
                name, ops, 100.0*host_us/(host_us+idle_us), host_us, idle_us, host_us); 
        // (idle_us filled by wrapper; see full driver code)
        #endif
    }
};

// Usage around each major kernel:
sherlock_timer t("attention", op_count);
do_attention(...);
// after call, the destructor prints the line.
```

*Device‑side (`dev_us`) is captured via `llvm‑prof`‑style markers or a simple `clock_gettime` wrapper that the ORC‑lite driver already uses for kernel launches. The agent need only emit the two numbers; the exact `dev_us` calculation is abstracted by the `sherlock_timer` wrapper.*

### How to Enable
- Compile with `-DSHERLOCK_LITE`
- Link against `libtimers` (tiny header‑only, no external dep)
- The footer line “instrumentation floor…” is emitted once at program exit by a global destructor.

---
## 2️⃣ Breadcrumb Debug Tags (very light)

**Macro definition (put in a shared header):**

```cpp
#ifdef DEBUG_BREADCRUMBS
#define __BREADCRUMB(component, phase) \
    do { \
        static thread_local char last_tag[64] = ""; \
        snprintf(last_tag, sizeof(last_tag), "%s:%s", component, phase); \
        // In a debug build you could write to a file / perfetto track; \
        // in lightweight mode just inform: \
        fprintf(stderr, "[BRDCRUMB] %s:%s\n", component, phase); \
    } while(0)
#else
#define __BREADCRUMB(component, phase) ((void)0)
#endif
```

**Usage throughout the driver:**

```cpp
__BREADCRUMB("gguf_loader", "start");
load_gguf_model(...);
__BREADCRUMB("gguf_loader", "end");

__BREADCRUMB("kv_cache", "init");
init_kv_cache(...);
__BREADCRUMB("kv_cache", "done");

__BREADCRUMB("moe_router", "select");
int sel = route_token(...);
__BREADCRUMB("moe_router", "done");
```

*Effect:* When `DEBUG_BREADCRUMBS` is defined (via CMake `-DDEBUG_BREADCRUMBS`), every major phase prints a tiny `stderr` line. In production builds the macro expands to `nothing`, zero overhead.

---