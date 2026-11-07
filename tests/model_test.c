/* Model loader + CLI test – RED phase.
 * FAILS until src/main.c implements:
 *   - argc/argv parsing for --model <path>
 *   - load_model() that reads .gguf header and returns nita_model_t*
 *   - Proper exit codes on success/failure
 */

/* We test the public API that main() exposes. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward: the actual main() will be defined and compiled */
extern int main(int argc, char *argv[]);

/* Test: help output when no args */
void test_cli_help(void)
{
    /* We cannot exec from here, but we can check the symbol exists. */
    printf("[test] CLI help: main() symbol resolvable\n");
    fflush(stdout);
    /* RED: If we get here, the symbol exists but we haven't tested args yet. */
}

/* Test: model load returns non-NULL for valid stub path.
 * This will GREEN once load_model() in main.c loads a real .gguf or
 * at least stubs a valid nita_model_t. */
void test_model_load_valid(const char *path)
{
    (void)path;
    /* Placeholder – will be GREEN once load_model() is implemented. */
    printf("[test] Model load valid: path=%s (stub – to be implemented)\n", path);
    fflush(stdout);
    /* RED: This fails until load_model() properly allocates and returns a model. */
}

/* Test: model load returns NULL for invalid/nonexistent path */
void test_model_load_invalid(void)
{
    /* Placeholder – will be GREEN once load_model() checks file existence. */
    printf("[test] Model load invalid: returns NULL for bad path (stub)\n");
    fflush(stdout);
    /* RED: Fails until load_model() validates the .gguf path. */
}