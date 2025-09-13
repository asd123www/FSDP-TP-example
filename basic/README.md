# NCCL Tuner Plugin with AllGather and ReduceScatter Optimization

This directory contains an implementation of an NCCL tuner plugin that optimizes the number of channels for AllGather and ReduceScatter collective operations based on their sequence in a typical training iteration. The plugin also prints detailed information about all collective operations being executed.

## Purpose

This plugin is designed to:
- Optimize AllGather operations by:
  - Using 16 channels for the first large AllGather (≥ 65537024 bytes)
  - Using 1 channel for the next 32 AllGather operations
  - Using 8 channels for the following 32 AllGather operations
- Optimize ReduceScatter operations by:
  - Using 1 channel for the first 28 ReduceScatter operations
  - Using 8 channels for the remaining ReduceScatter operations
- Track the sequence of collective operations to apply appropriate tuning
- Print detailed information about all collective operations (only from rank 0)
- Display algorithm, protocol, number of channels, message sizes, and other relevant information
- Provide a working example of the NCCL tuner plugin interface with custom tuning logic


## Implementation Details

The plugin implements the following functions:

### `pluginInit`
```c
ncclResult_t pluginInit(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context)
```
- **Purpose**: Initialize the plugin with communicator information
- **Current Implementation**: 
  - Allocates a context structure to store the logger function, rank info, and counters
  - Initializes counters for tracking AllGather and ReduceScatter operations
  - Determines current rank from environment variables
  - Only prints initialization message from rank 0
  - Displays tuning parameters that will be used
  - Stores rank and node information for later use
- **Parameters**:
  - `nRanks`: Total number of ranks in the communicator
  - `nNodes`: Total number of nodes in the communicator
  - `logFunction`: NCCL debug logging function
  - `context`: Plugin context pointer (output)

### `pluginGetCollInfo`
```c
ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels)
```
- **Purpose**: Apply tuning strategy and modify cost tables for collective operations
- **Current Implementation**:
  - Tracks the sequence of AllGather and ReduceScatter operations using counters
  - Dynamically sets the number of channels based on the collective type and sequence:
    - For AllGather: 16 channels for first large operation, then 1 channel for first 32 ops, 8 channels for next 32 ops
    - For ReduceScatter: 1 channel for first 28 operations, 8 channels for remaining operations
  - Prints detailed information about the collective operation being executed (only from rank 0)
  - Displays the collective type, message size, sequence number, and selected channel count
  - Shows the cost table with algorithm and protocol combinations
  - Sets RING+SIMPLE algorithm to cost 0.0 (highest preference) on all ranks
  - Uses stored rank information to determine whether to print
- **Parameters**:
  - `context`: Plugin context from init
  - `collType`: Type of collective operation
  - `nBytes`: Message size in bytes
  - `numPipeOps`: Number of pipeline operations
  - `collCostTable`: Cost table to modify
  - `numAlgo`: Number of algorithms
  - `numProto`: Number of protocols
  - `regBuff`: Whether buffer can be registered
  - `nChannels`: Number of channels to use (output)

### `pluginDestroy`
```c
ncclResult_t pluginDestroy(void* context)
```
- **Purpose**: Clean up plugin resources
- **Current Implementation**: 
  - Prints final counts of AllGather and ReduceScatter operations (only from rank 0)
  - Frees the allocated context structure
  - Prints a destruction message (only from rank 0)
  - Uses stored rank information to determine whether to print

## Channel Tuning Logic

The plugin implements a specific tuning strategy for AllGather and ReduceScatter operations:

### AllGather Tuning
```
if (collType == ncclFuncAllGather) {
  // Check if this is the first large AllGather
  if (nBytes >= LARGE_ALLGATHER_SIZE && !ctx->firstLargeAllGatherSeen) {
    *nChannels = ALLGATHER_HIGH_CHANNELS;  // 16 channels
    ctx->firstLargeAllGatherSeen = 1;
  } else if (ctx->allGatherCount < 32) {
    *nChannels = ALLGATHER_LOW_CHANNELS;   // 1 channel
  } else if (ctx->allGatherCount < 64) {
    *nChannels = ALLGATHER_MID_CHANNELS;   // 8 channels
  } else {
    *nChannels = ALLGATHER_LOW_CHANNELS;   // 1 channel
  }
  ctx->allGatherCount++;
}
```

### ReduceScatter Tuning
```
else if (collType == ncclFuncReduceScatter) {
  if (ctx->reduceScatterCount < 28) {
    *nChannels = REDUCESCATTER_LOW_CHANNELS;  // 1 channel
  } else {
    *nChannels = REDUCESCATTER_HIGH_CHANNELS; // 8 channels
  }
  ctx->reduceScatterCount++;
}
```

### Tuning Parameters
```c
#define LARGE_ALLGATHER_SIZE 65537024  // Size of the first large allgather (bytes)
#define ALLGATHER_HIGH_CHANNELS 16     // Number of channels for first allgather
#define ALLGATHER_LOW_CHANNELS 1       // Number of channels for first 32 subsequent allgathers
#define ALLGATHER_MID_CHANNELS 8       // Number of channels for next 32 allgathers
#define REDUCESCATTER_LOW_CHANNELS 1   // Number of channels for first 28 reduce-scatter ops
#define REDUCESCATTER_HIGH_CHANNELS 8  // Number of channels for last 5 reduce-scatter ops
```

## Cost Table Structure

The plugin demonstrates how to modify NCCL's cost tables:

```c
float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
```

The cost table is a 2D array where:
- First dimension: Algorithm index (e.g., `NCCL_ALGO_RING`)
- Second dimension: Protocol index (e.g., `NCCL_PROTO_SIMPLE`)
- Values: Cost for that algorithm/protocol combination

### Cost Values
- **0.0**: Highest preference (lowest cost)
- **Positive values**: Relative costs (lower is better)
- **`NCCL_ALGO_PROTO_IGNORE`**: Disable this combination

## Building

```bash
make
```

This creates `libnccl-tuner-basic.so` which can be loaded by NCCL.

## Usage

### Loading the Plugin

```bash
# Add plugin directory to library path
export LD_LIBRARY_PATH=/path/to/basic:$LD_LIBRARY_PATH

# Specify the tuner plugin to use (any of the following methods)
export NCCL_TUNER_PLUGIN=basic
# OR
export NCCL_TUNER_PLUGIN=libnccl-tuner-basic.so
# OR
export NCCL_TUNER_PLUGIN=/path/to/your/plugin/libnccl-tuner-basic.so

# Run your NCCL application
mpirun -np 4 your_nccl_application
```

### Verifying Plugin Loading and Channel Tuning

Enable NCCL debug output to see if the plugin is loaded:

```bash
export NCCL_DEBUG=INFO
```

You should see messages indicating the tuner plugin is being used:

```
[NCCL Tuner Plugin] Initialized tuner plugin with channel tuning for AllGather and ReduceScatter
[NCCL Tuner Plugin] Ranks: 4, Nodes: 1
[NCCL Tuner Plugin] Current rank: 0
[NCCL Tuner Plugin] Tuning parameters:
  - Large AllGather threshold: 65537024 bytes
  - First large AllGather: 16 channels
  - First 32 AllGather ops: 1 channels
  - Next 32 AllGather ops: 8 channels
  - First 28 ReduceScatter ops: 1 channels
  - Last 5 ReduceScatter ops: 8 channels
```

For AllGather operations, you should see output like:

```
===== NCCL Collective Operation Information =====
Collective Type: AllGather
AllGather Count: 1
Message Size: 65537024 bytes
Number of Pipeline Operations: 1
Number of Algorithms: 7
Number of Protocols: 3
Registered Buffer: Yes

Cost Table:
Algorithm | Protocol | Cost
------------------------------
Tree      | LL       | 23.4567
Tree      | LL128    | 12.3456
Ring      | Simple   | 0.0000
Selected Number of Channels: 16
==============================================
```

For ReduceScatter operations, you should see output like:

```
===== NCCL Collective Operation Information =====
Collective Type: ReduceScatter
ReduceScatter Count: 29
Message Size: 4194304 bytes
Number of Pipeline Operations: 1
Number of Algorithms: 7
Number of Protocols: 3
Registered Buffer: Yes

Cost Table:
Algorithm | Protocol | Cost
------------------------------
Tree      | LL       | 23.4567
Tree      | LL128    | 12.3456
Ring      | Simple   | 0.0000
Selected Number of Channels: 8
==============================================
```

### Validating the Channel Tuning

To verify that the channel tuning is working correctly:

1. Run a NCCL test that performs multiple AllGather and ReduceScatter operations
2. Check the output from the plugin to confirm:
   - The first large AllGather (≥ 65537024 bytes) uses 16 channels
   - The next 32 AllGather operations use 1 channel
   - The following 32 AllGather operations use 8 channels
   - The first 28 ReduceScatter operations use 1 channel
   - The remaining ReduceScatter operations use 8 channels

3. When the application completes, check the final operation counts:
   ```
   [NCCL Tuner Plugin] Destroying plugin
   [NCCL Tuner Plugin] Final AllGather count: 65
   [NCCL Tuner Plugin] Final ReduceScatter count: 33
   ```

The output provides detailed information about each collective operation executed, which helps confirm that the tuning logic is working as expected.

## Extending the Plugin

This basic plugin provides a foundation that you can extend:

### 1. Add Configuration Logic

Modify `pluginGetCollInfo` to implement your tuning strategy:

```c
__hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels) {
  // Your custom tuning logic here
  if (nBytes < 1024) {
    // Small message optimization
    table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 0.0;
  } else {
    // Large message optimization
    table[NCCL_ALGO_RING][NCCL_PROTO_LL128] = 0.0;
  }

  // Dynamic channel selection
  *nChannels = (nBytes > 1024*1024) ? 4 : 1;

  return ncclSuccess;
}
```

### 2. Add Context Management

Use the context pointer to store plugin state:

```c
struct pluginContext {
  int initialized;
  size_t nRanks;
  size_t nNodes;
  // Add your plugin-specific data here
};
```

### 3. Add File-Based Configuration

Read configuration from files, environment variables, or other sources.

### 4. Add Topology Awareness

Use the `nRanks` and `nNodes` parameters to implement topology-specific tuning.

## File Structure

```
basic/
├── README.md          # This file
├── plugin.c           # Plugin implementation
├── Makefile           # Build configuration
└── nccl/              # NCCL header files
    └── tuner.h        # Tuner plugin interface definitions
```

## Next Steps

1. **Understand the Interface**: Study the function signatures and parameters
2. **Implement Your Logic**: Add your tuning strategy to `pluginGetCollInfo`
3. **Test Thoroughly**: Verify your plugin works with different message sizes and topologies
4. **Add Error Handling**: Implement proper error checking and resource management
5. **Document Your Changes**: Update this README with your specific implementation details

## Comparison with Example Plugin

- **Basic Plugin**: Minimal implementation, good for learning and simple use cases
- **Example Plugin**: Full-featured CSV-based configuration system, good for production use

Choose the basic plugin if you want to:
- Learn the tuner plugin interface
- Implement simple, hardcoded tuning strategies
- Build a custom plugin from scratch

Choose the example plugin if you want:
- File-based configuration
- Complex tuning strategies
- Production-ready features

## Resources

- [Parent Directory README](../README.md) - General tuner plugin development guide
- [Example Plugin](../example/README.md) - Fully featured implementation

This basic plugin provides the foundation you need to start developing custom NCCL tuner plugins. Extend it with your specific tuning logic and requirements.
