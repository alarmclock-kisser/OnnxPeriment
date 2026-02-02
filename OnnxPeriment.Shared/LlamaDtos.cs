using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace OnnxPeriment.Shared
{
    public class LlamaModelLoadOptions
    {
        public string ModelPath { get; set; } = string.Empty;
        public bool UseCuda { get; set; } = true;
        public uint ContextSize { get; init; } = 4096;
        public int GpuLayerCount { get; init; } = 0;
    }

    public class LlamaGenerateOptions
    {
        public int MaxTokens { get; init; } = 2048;
        public float Temperature { get; init; } = 0.7f;
        public float TopP { get; init; } = 0.95f;
        public int TopK { get; init; } = 40;
        public int StreamChunkSize { get; init; } = 64;
        public int? Seed { get; init; }
        public IReadOnlyList<string>? StopSequences { get; init; }

        public bool AppendStatsToResponse { get; init; } = true;
    }

    public record LlamaChatMessage
    {
        public required string Role { get; init; }
        public required string Content { get; init; }
        public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
    }

    public record LlamaChatContext
    {
        public required string Name { get; init; }
        public List<LlamaChatMessage> Messages { get; init; } = new();
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
    }

    public record LlamaResponseStats
    {
        public int TokenCount { get; init; }
        public double ElapsedSeconds { get; init; }
        public double TokensPerSecond { get; init; }
    }

    public record LlamaGenerateRequest
    {
        public required string Prompt { get; init; }
        public string[]? Base64Images { get; init; }
        public LlamaGenerateOptions Options { get; init; } = new();
        public string? Context { get; init; }
    }

    public record LlamaStreamChunk
    {
        public required string Content { get; init; }
        public int Index { get; init; }
        public bool IsFinal { get; init; }
        public string? FinishReason { get; init; }
        public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
        public LlamaResponseStats? Stats { get; init; }
    }

    public record LlamaGenerateResult
    {
        public required string Content { get; init; }
        public int ChunkCount { get; init; }
        public string? FinishReason { get; init; }
        public LlamaResponseStats? Stats { get; init; }
    }

    public record LlamaBackendStatus
    {
        public required string Name { get; init; }
        public string? AssemblyName { get; init; }
        public bool AssemblyAvailable { get; init; }
        public string? NativeLibraryPath { get; init; }
        public bool NativeLibraryFound { get; init; }
        public bool NativeLibraryLoadable { get; init; }
        public string? Error { get; init; }

        public string ToString(bool singleLine = false)
        {
            var sb = new StringBuilder();
            sb.Append($"Backend: {this.Name}");

            sb.Append(singleLine ? "; " : Environment.NewLine);
            sb.Append($"Assembly: {(string.IsNullOrEmpty(this.AssemblyName) ? "missing" : this.AssemblyName)} (Available: {this.AssemblyAvailable})");

            var nativeLabel = string.IsNullOrEmpty(this.NativeLibraryPath)
                ? "missing"
                : singleLine ? Path.GetFileName(this.NativeLibraryPath) : this.NativeLibraryPath;

            sb.Append(singleLine ? "; " : Environment.NewLine);
            sb.Append($"Native Library: {nativeLabel} (Found: {this.NativeLibraryFound}, Loadable: {this.NativeLibraryLoadable})");

            if (!string.IsNullOrEmpty(this.Error))
            {
                sb.Append(singleLine ? "; " : Environment.NewLine);
                sb.Append($"Error: {this.Error}");
            }
            return sb.ToString();
        }
    }
}
