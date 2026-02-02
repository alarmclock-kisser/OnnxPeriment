using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace OnnxPeriment.Shared
{
    public record GenAiSettings
    {
        public int MaxLength { get; init; } = 2048;
        public float Temperature { get; init; } = 0.0f; // 0.0 = Deterministic/OCR Modus
        public float TopP { get; init; } = 0.9f;
        public int TopK { get; init; } = 40;
        public bool DoSample { get; init; } = false; // Bei OCR meistens false
    }

    public record OnnxBackendStatus
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
