/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

 #include "tuner.h"
 #include <stdio.h>
 
 #define __hidden __attribute__ ((visibility("hidden")))
 
 // Context to store the logger function and rank information
 typedef struct {
   ncclDebugLogger_t logger;
   int rank;                  // Current rank
   size_t nRanks;             // Total number of ranks
   size_t nNodes;             // Total number of nodes
   int allGatherCount;        // Counter for AllGather operations
   int reduceScatterCount;    // Counter for ReduceScatter operations
   int firstLargeAllGatherSeen; // Flag to track if we've seen the first large AllGather
 } pluginContext_t;
 
 // Tuning parameters
 #define LARGE_ALLGATHER_SIZE 65537024 * 8  // Size of the first large allgather (bytes)
 #define ALLGATHER_HIGH_CHANNELS 24     // Number of channels for first allgather
 #define ALLGATHER_LOW_CHANNELS 1       // Number of channels for first 32 subsequent allgathers
 #define ALLGATHER_MID_CHANNELS 24       // Number of channels for next 32 allgathers
 #define REDUCESCATTER_LOW_CHANNELS 16   // Number of channels for first 28 reduce-scatter ops
 #define REDUCESCATTER_HIGH_CHANNELS 24  // Number of channels for last 5 reduce-scatter ops
 
 // Function to convert collective type to string
 const char* getCollectiveTypeString(ncclFunc_t collType) {
   switch(collType) {
     case ncclFuncBroadcast: return "Broadcast";
     case ncclFuncReduce: return "Reduce";
     case ncclFuncAllGather: return "AllGather";
     case ncclFuncReduceScatter: return "ReduceScatter";
     case ncclFuncAllReduce: return "AllReduce";
     case ncclFuncSendRecv: return "SendRecv";
     case ncclFuncSend: return "Send";
     case ncclFuncRecv: return "Recv";
     default: return "Unknown";
   }
 }
 
 // Function to convert algorithm type to string
 const char* getAlgorithmString(int algo) {
   switch(algo) {
     case NCCL_ALGO_TREE: return "Tree";
     case NCCL_ALGO_RING: return "Ring";
     case NCCL_ALGO_COLLNET_DIRECT: return "CollNet Direct";
     case NCCL_ALGO_COLLNET_CHAIN: return "CollNet Chain";
     case NCCL_ALGO_NVLS: return "NVLS";
     case NCCL_ALGO_NVLS_TREE: return "NVLS Tree";
     case NCCL_ALGO_PAT: return "PAT";
     default: return "Unknown";
   }
 }
 
 // Function to convert protocol type to string
 const char* getProtocolString(int proto) {
   switch(proto) {
     case NCCL_PROTO_LL: return "LL";
     case NCCL_PROTO_LL128: return "LL128";
     case NCCL_PROTO_SIMPLE: return "Simple";
     default: return "Unknown";
   }
 }
 
 __hidden ncclResult_t pluginInit(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context) {
   // Allocate and initialize our context
   pluginContext_t *ctx = (pluginContext_t*)malloc(sizeof(pluginContext_t));
   if (ctx == NULL) return ncclSystemError;
   
   // Store information in context
   ctx->logger = logFunction;
   ctx->nRanks = nRanks;
   ctx->nNodes = nNodes;
   
   // Initialize counters
   ctx->allGatherCount = 0;
   ctx->reduceScatterCount = 0;
   ctx->firstLargeAllGatherSeen = 0;
   
   // Get current rank from environment variable
   ctx->rank = -1;  // Default to -1 if not found
   char* rankStr = getenv("PMIX_RANK");  // Try PMIX first (common in MPI environments)
   if (!rankStr) rankStr = getenv("OMPI_COMM_WORLD_RANK");  // OpenMPI
   if (!rankStr) rankStr = getenv("PMI_RANK");  // MPICH
   if (!rankStr) rankStr = getenv("SLURM_PROCID");  // SLURM
   if (!rankStr) rankStr = getenv("MV2_COMM_WORLD_RANK");  // MVAPICH2
   if (!rankStr) rankStr = getenv("MPI_LOCALRANKID");  // Intel MPI
   if (!rankStr) rankStr = getenv("LOCAL_RANK");  // Generic
   if (rankStr) ctx->rank = atoi(rankStr);
   
   // Print initialization message only from rank 0 or if rank is unknown
   if (ctx->rank == 0 || ctx->rank == -1) {
     // printf("[NCCL Tuner Plugin] Initialized tuner plugin with channel tuning for AllGather and ReduceScatter\n");
     // printf("[NCCL Tuner Plugin] Ranks: %zu, Nodes: %zu\n", nRanks, nNodes);
     // printf("[NCCL Tuner Plugin] Current rank: %d\n", ctx->rank);
     // printf("[NCCL Tuner Plugin] Tuning parameters:\n");
     // printf("  - Large AllGather threshold: %d bytes\n", LARGE_ALLGATHER_SIZE);
     // printf("  - First large AllGather: %d channels\n", ALLGATHER_HIGH_CHANNELS);
     // printf("  - First 32 AllGather ops: %d channels\n", ALLGATHER_LOW_CHANNELS);
     // printf("  - Next 32 AllGather ops: %d channels\n", ALLGATHER_MID_CHANNELS);
     // printf("  - First 28 ReduceScatter ops: %d channels\n", REDUCESCATTER_LOW_CHANNELS);
     // printf("  - Last 5 ReduceScatter ops: %d channels\n", REDUCESCATTER_HIGH_CHANNELS);
   }
   
   *context = ctx;
   return ncclSuccess;
 }
 
 __hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                               int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                               int regBuff, int* nChannels) {
   // Get context
   pluginContext_t* ctx = (pluginContext_t*)context;
   if (!ctx) return ncclInternalError;
   
   // Update NCCL core generated cost table
   float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
   
   // Tune the number of channels based on collective type and sequence
   if (collType == ncclFuncAllGather) {
     // A new iteration start with a special message size.
     if (nBytes == LARGE_ALLGATHER_SIZE) {
       *nChannels = ALLGATHER_HIGH_CHANNELS;
       ctx->allGatherCount = 0;
       ctx->reduceScatterCount = 0;
       ctx->firstLargeAllGatherSeen = 1;
     } else if (ctx->allGatherCount < 32) {
       *nChannels = ALLGATHER_LOW_CHANNELS;
     } else if (ctx->allGatherCount < 64) {
       *nChannels = ALLGATHER_MID_CHANNELS;
     } else {
       // For any additional AllGather operations
       // *nChannels = ALLGATHER_LOW_CHANNELS;
     }
     ctx->allGatherCount++;
   } else if (collType == ncclFuncReduceScatter) {
     if (ctx->reduceScatterCount < 28) {
       *nChannels = REDUCESCATTER_LOW_CHANNELS;
     } else {
       *nChannels = REDUCESCATTER_HIGH_CHANNELS;
     }
     ctx->reduceScatterCount++;
   } else {
     // Default channel count for other collectives
     *nChannels = 1;
   }
   
   // Set RING+SIMPLE algorithm to highest preference (lowest cost)
   if (table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] != NCCL_ALGO_PROTO_IGNORE) {
     table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 0.0;
   }
   
   // Only print from rank 0 or if rank is unknown
   if (ctx && (ctx->rank == 0 || ctx->rank == -1)) {
     // Print collective information
     if (collType == ncclFuncAllGather && nBytes == LARGE_ALLGATHER_SIZE) {
       printf("The beginning of a iteration.\n");
     }
     // printf("===== NCCL Collective Operation Information =====\n");
     // printf("Collective Type: %s\n", getCollectiveTypeString(collType));
     
     if (collType == ncclFuncAllGather) {
       // printf("AllGather Count: %d\n", ctx->allGatherCount);
     } else if (collType == ncclFuncReduceScatter) {
       // printf("ReduceScatter Count: %d\n", ctx->reduceScatterCount);
     }
     
     // printf("Message Size: %zu bytes\n", nBytes);
     // printf("Number of Pipeline Operations: %d\n", numPipeOps);
     // printf("Number of Algorithms: %d\n", numAlgo);
     // printf("Number of Protocols: %d\n", numProto);
     // printf("Registered Buffer: %s\n", regBuff ? "Yes" : "No");
     
     // Print cost table
     // printf("\nCost Table:\n");
     // printf("Algorithm | Protocol | Cost\n");
     // printf("------------------------------\n");
     for (int a = 0; a < numAlgo; a++) {
       for (int p = 0; p < numProto; p++) {
         if (table[a][p] != NCCL_ALGO_PROTO_IGNORE) {
           // printf("%-10s | %-8s | %.4f\n", 
                 //  getAlgorithmString(a), 
                 //  getProtocolString(p), 
                 //  table[a][p]);
         }
       }
     }
     
     // printf("Selected Number of Channels: %d\n", *nChannels);
     // printf("==============================================\n\n");
   }
   
   return ncclSuccess;
 }
 
 __hidden ncclResult_t pluginDestroy(void* context) {
   if (context) {
     pluginContext_t* ctx = (pluginContext_t*)context;
     // Only print from rank 0 or if rank is unknown
     if (ctx->rank == 0 || ctx->rank == -1) {
       // printf("[NCCL Tuner Plugin] Destroying plugin\n");
       // printf("[NCCL Tuner Plugin] Final AllGather count: %d\n", ctx->allGatherCount);
       // printf("[NCCL Tuner Plugin] Final ReduceScatter count: %d\n", ctx->reduceScatterCount);
     }
     free(context);
   }
   return ncclSuccess;
 }
 
 #define PLUGIN_NAME "TuningPlugin"
 
 const ncclTuner_v4_t ncclTunerPlugin_v4 = {
   .name = PLUGIN_NAME,
   .init = pluginInit,
   .getCollInfo = pluginGetCollInfo,
   .destroy = pluginDestroy
 };