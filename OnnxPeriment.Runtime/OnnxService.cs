using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntimeGenAI;
using OnnxPeriment.Core;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.ConstrainedExecution;
using System.Text;
using SixLabors.ImageSharp;
using OnnxPeriment.Shared;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace OnnxPeriment.Runtime
{
    public class OnnxService : IDisposable
    {
        private InferenceSession? _session;
        public string? LoadedModelPath { get; private set; } = null;

        // GenAI spezifische Felder
        private Model? _genAiModel;
        private Tokenizer? _genAiTokenizer;
        private MultiModalProcessor? _genAiProcessor;
        public bool ModelIsLoaded => this._genAiModel != null;

        public ModelMetadata? LoadedModelMetadata => this._session?.ModelMetadata;
        public Dictionary<string, string>? LoadedModelMetadataMap => this._session?.ModelMetadata.CustomMetadataMap;

        public List<string> ModelDirectories { get; private set; } =
        [
            "D:\\Models",
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), ".lmstudio", "models"),
            "D:\\Programme\\LLM_MODELS"
        ];

        public List<string> ModelPaths { get; private set; } = [];

        public OnnxService()
        {
            this.ModelPaths = this.GetAvailableModelPaths();
        }

        private sealed class ManagedContext
        {
            public required LlamaChatContext Context { get; init; }
            public bool AutoSave { get; init; }
            public bool IsTransient { get; init; }
        }

        private const string RecentContextName = "recent";
        private const string DefaultContextName = "default";

        private readonly Dictionary<string, LlamaChatContext> _contexts = new(StringComparer.OrdinalIgnoreCase);
        private LlamaChatContext? _activeContext;

        public string? LoadedContextFile { get; private set; }

        // I/O
        public List<string> GetAvailableModelPaths(string[]? additionalDirectories = null, bool includeSubDirectories = true)
        {
            List<string> dirs = new List<string>(this.ModelDirectories);
            if (additionalDirectories != null)
            {
                dirs.AddRange(additionalDirectories);
            }

            List<string> modelPaths = dirs
                .Where(Directory.Exists)
                .SelectMany(dir => Directory.GetFiles(dir, "*.onnx", includeSubDirectories ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly))
                .ToList();

            StaticLogger.Log($"Found {modelPaths.Count} ONNX model(s).");
            this.ModelPaths = modelPaths;
            return modelPaths;
        }

        public List<OnnxBackendStatus> VerifyBackends(bool testLoad = true)
        {
            var searchPaths = new List<string>
            {
                AppContext.BaseDirectory,
                Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64", "native")
            };

            searchPaths = searchPaths
                .Where(path => Directory.Exists(path))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            var probes = new[]
            {
                new
                {
                    Name = "CPU",
                    Assemblies = new[] { "Microsoft.ML.OnnxRuntime" },
                    NativeCandidates = new[] { "onnxruntime.dll" }
                },
                new
                {
                    Name = "CUDA",
                    Assemblies = new[] { "Microsoft.ML.OnnxRuntime.Gpu", "Microsoft.ML.OnnxRuntime.Gpu.Windows" },
                    NativeCandidates = new[] { "onnxruntime_providers_cuda.dll", "onnxruntime.dll" }
                }
            };

            var results = new List<OnnxBackendStatus>();

            foreach (var probe in probes)
            {
                string? loadedAssemblyName = null;
                bool assemblyAvailable = false;
                string? error = null;

                foreach (var assemblyName in probe.Assemblies)
                {
                    try
                    {
                        Assembly.Load(assemblyName);
                        loadedAssemblyName = assemblyName;
                        assemblyAvailable = true;
                        break;
                    }
                    catch
                    {
                        // ignore and try next
                    }
                }

                string? nativePath = null;
                foreach (var candidate in probe.NativeCandidates)
                {
                    nativePath = searchPaths
                        .Select(path => Path.Combine(path, candidate))
                        .FirstOrDefault(File.Exists);
                    if (nativePath != null)
                    {
                        break;
                    }
                }

                bool nativeFound = nativePath != null;
                bool nativeLoadable = false;

                if (testLoad && nativePath != null)
                {
                    try
                    {
                        if (NativeLibrary.TryLoad(nativePath, out var handle))
                        {
                            nativeLoadable = true;
                            NativeLibrary.Free(handle);
                        }
                    }
                    catch (Exception ex)
                    {
                        error = ex.Message;
                    }
                }

                results.Add(new OnnxBackendStatus
                {
                    Name = probe.Name,
                    AssemblyName = loadedAssemblyName,
                    AssemblyAvailable = assemblyAvailable,
                    NativeLibraryPath = nativePath,
                    NativeLibraryFound = nativeFound,
                    NativeLibraryLoadable = nativeLoadable,
                    Error = error
                });
            }

            foreach (var result in results)
            {
                StaticLogger.Log($"ONNX backend {result.Name}: assembly={(result.AssemblyAvailable ? result.AssemblyName : "missing")}, native={(result.NativeLibraryFound ? result.NativeLibraryPath : "missing")}, loadable={result.NativeLibraryLoadable}");
                if (!string.IsNullOrWhiteSpace(result.Error))
                {
                    StaticLogger.Log($"ONNX backend {result.Name} error: {result.Error}");
                }
            }

            return results;
        }

        public async Task<IReadOnlyList<string>> EnsureCudaDependenciesAsync(string? cudaRoot = null, string? outputDir = null)
        {
            return await Task.Run(() =>
            {
                var logs = new List<string>();

                string? resolvedCudaRoot = ResolveCudaRoot(cudaRoot);
                if (string.IsNullOrWhiteSpace(resolvedCudaRoot))
                {
                    logs.Add("CUDA root not found. Set CUDA_PATH or ensure nvcc is in PATH.");
                    return logs;
                }

                var cudaBin = Path.Combine(resolvedCudaRoot, "bin");
                if (!Directory.Exists(cudaBin))
                {
                    logs.Add($"CUDA bin folder not found: {cudaBin}");
                    return logs;
                }

                var targetDir = outputDir ?? Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64", "native");
                Directory.CreateDirectory(targetDir);

                var cudaDlls = new[]
                {
                    "cudart64_12.dll",
                    "cublas64_12.dll",
                    "cublasLt64_12.dll",
                    "cufft64_12.dll",
                    "curand64_10.dll",
                    "cusolver64_11.dll",
                    "cusparse64_12.dll"
                };

                var cudnnDlls = new[]
                {
                    "cudnn64_9.dll",
                    "cudnn_adv64_9.dll",
                    "cudnn_ops64_9.dll",
                    "cudnn_cnn64_9.dll"
                };

                CopyDlls(cudaBin, targetDir, cudaDlls, logs, "CUDA");
                CopyDlls(cudaBin, targetDir, cudnnDlls, logs, "cuDNN", warnWhenMissing: false);

                var providerPath = Path.Combine(targetDir, "onnxruntime_providers_cuda.dll");
                if (!File.Exists(providerPath))
                {
                    logs.Add("onnxruntime_providers_cuda.dll not found in output. Ensure OnnxRuntime.Gpu package is referenced and rebuilt.");
                }

                logs.Add($"CUDA root: {resolvedCudaRoot}");
                logs.Add($"Output dir: {targetDir}");
                return logs;
            });
        }

        private static string? ResolveCudaRoot(string? cudaRoot)
        {
            if (!string.IsNullOrWhiteSpace(cudaRoot) && Directory.Exists(cudaRoot))
            {
                return Path.GetFullPath(cudaRoot);
            }

            var envCuda = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (!string.IsNullOrWhiteSpace(envCuda) && Directory.Exists(envCuda))
            {
                return envCuda;
            }

            var path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            foreach (var entry in path.Split(Path.PathSeparator))
            {
                var nvcc = Path.Combine(entry.Trim(), "nvcc.exe");
                if (File.Exists(nvcc))
                {
                    return Directory.GetParent(entry.Trim())?.FullName;
                }
            }

            var defaultPath = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8";
            if (Directory.Exists(defaultPath))
            {
                return defaultPath;
            }

            return null;
        }

        private static void CopyDlls(string sourceDir, string targetDir, string[] dlls, List<string> logs, string label, bool warnWhenMissing = true)
        {
            foreach (var dll in dlls)
            {
                var src = Path.Combine(sourceDir, dll);
                var dest = Path.Combine(targetDir, dll);
                if (File.Exists(src))
                {
                    File.Copy(src, dest, true);
                    logs.Add($"{label}: copied {dll}");
                }
                else if (warnWhenMissing)
                {
                    logs.Add($"{label}: missing {dll}");
                }
            }
        }

        private static bool TryGetInputIdsLength(string? message, out int inputIdsLength)
        {
            inputIdsLength = 0;
            if (string.IsNullOrWhiteSpace(message))
            {
                return false;
            }

            const string marker = "input_ids size (";
            var start = message.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
            if (start < 0)
            {
                return false;
            }

            start += marker.Length;
            var end = message.IndexOf(')', start);
            if (end <= start)
            {
                return false;
            }

            return int.TryParse(message.AsSpan(start, end - start), out inputIdsLength);
        }
        // Loads GenAI Model fallback to cpu
        public async Task<string?> LoadGenAiModelAsync(string modelPathOrDirectory, bool useCuda = true)
        {
            return await Task.Run(async () =>
            {
                try
                {
                    // 1. Pfad validieren
                    string directoryPath = Directory.Exists(modelPathOrDirectory)
                        ? modelPathOrDirectory
                        : Path.GetDirectoryName(modelPathOrDirectory)!;

                    if (string.IsNullOrEmpty(directoryPath) || !Directory.Exists(directoryPath))
                    {
                        string err = "Invalid GenAI model directory.";
                        await StaticLogger.LogAsync(err);
                        return err;
                    }

                    // Ressourcen freigeben
                    this.DisposeGenAiResources();

                    // 2. Ladeversuch mit CUDA (falls gewünscht)
                    if (useCuda)
                    {
                        try
                        {
                            await StaticLogger.LogAsync("Attempting to load GenAI model with CUDA...");
                            // Hinweis: Das GenAI-Modell entscheidet oft über das installierte NuGet (Cuda vs Cpu)
                            // Falls du das .Cuda NuGet hast, wird CUDA bevorzugt.
                            this._genAiModel = new Model(directoryPath);
                            await StaticLogger.LogAsync("GenAI model loaded successfully with CUDA.");
                        }
                        catch (Exception ex)
                        {
                            await StaticLogger.LogAsync($"CUDA Load failed, falling back to CPU: {ex.Message}");
                            this._genAiModel = null; // Sicherstellen, dass wir sauber neu starten
                        }
                    }

                    // 3. Fallback auf CPU (wird ausgeführt wenn useCuda=false oder CUDA-Laden fehlgeschlagen ist)
                    if (this._genAiModel == null)
                    {
                        await StaticLogger.LogAsync("Loading GenAI model with CPU...");
                        // Bei der GenAI Runtime wird das Device oft über die Library-Version gesteuert.
                        // Im Falle eines harten CPU-Fallbacks müssten ggf. Environment-Variablen gesetzt werden,
                        // meistens lädt die Model-Klasse aber automatisch das verfügbare Backend.
                        this._genAiModel = new Model(directoryPath);
                        await StaticLogger.LogAsync("GenAI model loaded on CPU.");
                    }

                    // 4. Tokenizer und Processor initialisieren
                    this._genAiTokenizer = new Tokenizer(this._genAiModel);

                    // Prüfen ob es ein multimodales Modell ist (wie Phi-3.5 Vision)
                    try
                    {
                        this._genAiProcessor = new MultiModalProcessor(this._genAiModel);
                        await StaticLogger.LogAsync("Multimodal processor initialized (Vision support enabled).");
                    }
                    catch
                    {
                        await StaticLogger.LogAsync("Model does not appear to be multimodal. Standard text-gen only.");
                    }

                    await StaticLogger.LogAsync("GenAI model loading process completed.");
                    await StaticLogger.LogAsync($"Graph name: {this.LoadedModelMetadata?.GraphName}");
                    await StaticLogger.LogAsync($"Custom metadata [{this.LoadedModelMetadataMap?.Count}]:");
                    if (this.LoadedModelMetadataMap != null)
                    {
                        foreach (var kvp in this.LoadedModelMetadataMap)
                        {
                            await StaticLogger.LogAsync($"  {kvp.Key}: {kvp.Value}");
                        }
                    }   
                    this.LoadedModelPath = Path.GetFullPath(directoryPath);
                    return null; // Erfolg
                }
                catch (Exception ex)
                {
                    await StaticLogger.LogAsync(ex);
                    return ex.Message;
                }
            });
        }


        public void UnloadModel()
        {
            this.DisposeGenAiResources();
            this.LoadedModelPath = null;
        }


        // Run GenAI Inference OCR Prompt
        public async Task<string> RunOcrPromptAsync(string base64Image, string userPrompt, GenAiSettings? settings = null)
        {
            if (this._genAiModel == null || this._genAiProcessor == null)
            {
                return "Modell nicht geladen.";
            }

            settings ??= new GenAiSettings();

            return await Task.Run(async () =>
            {
                string tempFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".png");
                string normalizedTemp = tempFile;
                try
                {
                    // Bild speichern
                    if (string.IsNullOrWhiteSpace(base64Image))
                    {
                        await StaticLogger.LogAsync("No image data provided for OCR request.");
                    }

                    await File.WriteAllBytesAsync(tempFile, Convert.FromBase64String(base64Image));

                    // Diagnostic: log file size and header bytes
                    try
                    {
                        var fileInfo = new FileInfo(tempFile);
                        var fileSize = fileInfo.Length;
                        byte[] header = new byte[Math.Min(16, fileSize)];
                        using (var fs = File.OpenRead(tempFile))
                        {
                            await fs.ReadExactlyAsync(header);
                        }
                        string headerHex = BitConverter.ToString(header).Replace("-", " ");
                        bool looksLikePng = header.Length >= 8 && header[0] == 0x89 && header[1] == 0x50 && header[2] == 0x4E && header[3] == 0x47 && header[4] == 0x0D && header[5] == 0x0A && header[6] == 0x1A && header[7] == 0x0A;
                        await StaticLogger.LogAsync($"Wrote temp image: {tempFile} ({fileSize} bytes), header: {headerHex}, looksLikePng={looksLikePng}");
                    }
                    catch (Exception ex)
                    {
                        await StaticLogger.LogAsync($"Failed to read temp image header: {ex.Message}");
                    }

                    // Validate + normalize using ImageSharp to ensure compatible PNG container
                    try
                    {
                        using var img = SixLabors.ImageSharp.Image.Load(tempFile);
                        await StaticLogger.LogAsync($"Temp image loaded for normalization: {img.Width}x{img.Height}, format: {img.Metadata.DecodedImageFormat?.Name}");

                        var norm = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".png");
                        await img.SaveAsPngAsync(norm);
                        // replace tempFile with normalized file
                        try { File.Delete(tempFile); } catch { }
                        normalizedTemp = norm;
                        await StaticLogger.LogAsync($"Image normalized and saved to: {normalizedTemp} (size {new FileInfo(normalizedTemp).Length} bytes)");
                    }
                    catch (Exception ex)
                    {
                        await StaticLogger.LogAsync($"Image normalization failed: {ex.Message}");
                        // keep original tempFile; proceed to let native decoder attempt
                    }

                    // 1. Bild als Array laden (string[] erforderlich)
                    using var images = Images.Load([normalizedTemp]);

                    // 2. Prompt für Phi-3.5 Vision
                    string fullPrompt = $"<|user|>\n<|image_1|>\n{userPrompt}<|end|>\n<|assistant|>\n";

                    // 3. Tensors erzeugen
                    using var inputTensors = this._genAiProcessor.ProcessImages(fullPrompt, images);

                    // 4. Generator konfigurieren
                    int maxLength = settings.MaxLength;
                    int maxNewTokens = Math.Max(1, settings.MaxNewTokens);

                    GeneratorParams? generatorParams = null;
                    Generator? generator = null;

                    try
                    {
                        generatorParams = new GeneratorParams(this._genAiModel);
                        generatorParams.SetSearchOption("max_length", maxLength);
                        generatorParams.SetSearchOption("do_sample", settings.DoSample);

                        // 5. Generator erzeugen und Inputs setzen
                        generator = new Generator(this._genAiModel, generatorParams);

                        // HIER: SetInputs direkt am Generator (wie von dir entdeckt)
                        try
                        {
                            generator.SetInputs(inputTensors);
                        }
                        catch (Exception ex) when (TryGetInputIdsLength(ex.Message, out var inputIdsLength))
                        {
                            int adjustedMaxLength = Math.Max(maxLength, inputIdsLength + maxNewTokens);
                            if (adjustedMaxLength != maxLength)
                            {
                                await StaticLogger.LogAsync($"Adjusting max_length from {maxLength} to {adjustedMaxLength} to fit input size {inputIdsLength}.");
                                maxLength = adjustedMaxLength;
                                generator.Dispose();
                                generatorParams.Dispose();

                                generatorParams = new GeneratorParams(this._genAiModel);
                                generatorParams.SetSearchOption("max_length", maxLength);
                                generatorParams.SetSearchOption("do_sample", settings.DoSample);

                                generator = new Generator(this._genAiModel, generatorParams);
                                generator.SetInputs(inputTensors);
                            }
                            else
                            {
                                throw;
                            }
                        }

                        StringBuilder sb = new StringBuilder();

                        while (!generator.IsDone())
                        {
                            // In der neuen API reicht oft GenerateNextToken(), 
                            // da es die Logits-Berechnung implizit triggert.
                            generator.GenerateNextToken();

                            // Letztes Token aus der Sequenz holen
                            var sequence = generator.GetSequence(0).ToArray();
                            if (sequence.Length > 0)
                            {
                                var lastToken = sequence[^1..]; // Letztes Element
                                string piece = this._genAiTokenizer!.Decode(lastToken);
                                sb.Append(piece);

                                // Optional: Debugging im Logger
                                // StaticLogger.Log($"Token: {piece}");
                            }
                        }

                        return sb.ToString();
                    }
                    finally
                    {
                        generator?.Dispose();
                        generatorParams?.Dispose();
                    }
                }
                catch (Exception ex)
                {
                    await StaticLogger.LogAsync(ex);
                    return $"Fehler im Loop: {ex.Message}";
                }
                finally
                {
                    try { if (File.Exists(tempFile)) File.Delete(tempFile); } catch { }
                    try { if (!string.IsNullOrEmpty(normalizedTemp) && File.Exists(normalizedTemp)) File.Delete(normalizedTemp); } catch { }
                }
            });
        }





        // Dispose
        private void DisposeGenAiResources()
        {
            this._genAiProcessor?.Dispose();
            this._genAiTokenizer?.Dispose();
            this._genAiModel?.Dispose();
            this._genAiProcessor = null;
            this._genAiTokenizer = null;
            this._genAiModel = null;
        }

        public void Dispose()
        {
            this._session?.Dispose();
            this.DisposeGenAiResources();
        }

        public async Task<IReadOnlyList<string>> GetContextsAsync()
        {
            return await Task.Run(() =>
            {
                var dir = GetContextDirectory();
                Directory.CreateDirectory(dir);

                return Directory.EnumerateFiles(dir, "*.json")
                    .Select(path => new FileInfo(path))
                    .OrderByDescending(info => info.LastWriteTimeUtc)
                    .Select(info => Path.GetFileNameWithoutExtension(info.Name))
                    .ToList();
            });
        }

        public async Task<LlamaChatContext?> LoadContextAsync(string nameOrRecent)
        {
            if (string.IsNullOrWhiteSpace(nameOrRecent))
            {
                return null;
            }

            return await Task.Run(() =>
            {
                string? resolved;
                if (File.Exists(nameOrRecent))
                {
                    resolved = Path.GetFullPath(nameOrRecent);
                }
                else
                {
                    var normalized = NormalizeContextName(nameOrRecent);
                    if (string.IsNullOrWhiteSpace(normalized))
                    {
                        return null;
                    }

                    resolved = ResolveContextFilePath(normalized);
                }

                if (string.IsNullOrWhiteSpace(resolved) || !File.Exists(resolved))
                {
                    return null;
                }

                var context = LoadContextFromFile(resolved);
                if (context == null)
                {
                    return null;
                }

                _contexts[context.Name] = context;
                this.LoadedContextFile = resolved;
                this._activeContext = context;
                return context;
            });
        }

        public async Task<LlamaChatContext> CreateContextAsync(string name = "")
        {
            return await Task.Run(() =>
            {
                var context = new LlamaChatContext
                {
                    Name = string.IsNullOrWhiteSpace(name) ? GenerateTransientContextName() : name,
                    UpdatedAt = DateTime.UtcNow
                };

                if (!string.IsNullOrWhiteSpace(name))
                {
                    _contexts[context.Name] = context;
                }

                this._activeContext = context;
                return context;
            });
        }

        public async Task<bool> SaveContextAsync(string? name = null)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                name = this._activeContext?.Name;
            }
            if (string.IsNullOrWhiteSpace(name))
            {
                return false;
            }

            if (!_contexts.TryGetValue(name, out var context))
            {
                return false;
            }

            return await SaveContextAsync(context);
        }

        public async Task<bool> SaveContextAsync(LlamaChatContext context)
        {
            if (context == null || string.IsNullOrWhiteSpace(context.Name))
            {
                return false;
            }

            return await Task.Run(() =>
            {
                var dir = GetContextDirectory();
                Directory.CreateDirectory(dir);
                var safeName = SanitizeContextName(context.Name);
                var path = Path.Combine(dir, safeName + ".json");
                context.UpdatedAt = DateTime.UtcNow;

                var json = JsonSerializer.Serialize(context, new JsonSerializerOptions
                {
                    WriteIndented = true
                });
                File.WriteAllText(path, json, Encoding.UTF8);
                _contexts[context.Name] = context;
                this.LoadedContextFile = path;
                this._activeContext = context;
                return true;
            });
        }

        private LlamaChatContext? ResolveActiveContext()
        {
            if (this._activeContext != null)
            {
                return this._activeContext;
            }

            if (!string.IsNullOrWhiteSpace(this.LoadedContextFile) && File.Exists(this.LoadedContextFile))
            {
                var loaded = LoadContextFromFile(this.LoadedContextFile);
                if (loaded != null)
                {
                    this._activeContext = loaded;
                    _contexts[loaded.Name] = loaded;
                    return loaded;
                }
            }

            return _contexts.Values
                .OrderByDescending(context => context.UpdatedAt)
                .FirstOrDefault();
        }

        public int GetContextMessagePairCount()
        {
            var context = ResolveActiveContext();
            if (context == null)
            {
                return 0;
            }

            var count = context.Messages?.Count ?? 0;
            return (count + 1) / 2;
        }

        public List<LlamaChatMessage> GetContextMessagesPair(int? index = null)
        {
            var context = ResolveActiveContext();
            if (context == null)
            {
                return [];
            }

            var allMessages = context.Messages ?? [];
            if (index == null)
            {
                return [.. allMessages];
            }

            var pairIndex = index.Value;
            if (pairIndex <= 0)
            {
                return [];
            }

            int start = (pairIndex - 1) * 2;
            if (start >= allMessages.Count)
            {
                return [];
            }

            var result = new List<LlamaChatMessage> { allMessages[start] };
            if (start + 1 < allMessages.Count)
            {
                result.Add(allMessages[start + 1]);
            }

            return result;
        }

        private async Task<ManagedContext?> ResolveContextAsync(string? contextName)
        {
            if (contextName == null)
            {
                var active = ResolveActiveContext();
                if (active != null)
                {
                    var isTransient = !_contexts.ContainsKey(active.Name);
                    var autoSave = !string.IsNullOrWhiteSpace(this.LoadedContextFile) && File.Exists(this.LoadedContextFile);
                    return new ManagedContext
                    {
                        Context = active,
                        AutoSave = autoSave,
                        IsTransient = isTransient
                    };
                }

                var transient = new LlamaChatContext
                {
                    Name = GenerateTransientContextName(),
                    UpdatedAt = DateTime.UtcNow
                };
                this._activeContext = transient;
                return new ManagedContext
                {
                    Context = transient,
                    AutoSave = false,
                    IsTransient = true
                };
            }

            var normalized = NormalizeContextName(contextName);
            if (string.IsNullOrWhiteSpace(normalized))
            {
                var transient = new LlamaChatContext
                {
                    Name = GenerateTransientContextName(),
                    UpdatedAt = DateTime.UtcNow
                };
                this._activeContext = transient;
                return new ManagedContext
                {
                    Context = transient,
                    AutoSave = false,
                    IsTransient = true
                };
            }

            if (string.Equals(normalized, RecentContextName, StringComparison.OrdinalIgnoreCase)
                || string.Equals(normalized, DefaultContextName, StringComparison.OrdinalIgnoreCase))
            {
                var resolved = ResolveContextFilePath(normalized);
                if (resolved != null && File.Exists(resolved))
                {
                    var loaded = LoadContextFromFile(resolved);
                    if (loaded != null)
                    {
                        _contexts[loaded.Name] = loaded;
                        this._activeContext = loaded;
                        return new ManagedContext { Context = loaded, AutoSave = true, IsTransient = false };
                    }
                }

                var fallback = new LlamaChatContext
                {
                    Name = DefaultContextName,
                    UpdatedAt = DateTime.UtcNow
                };
                _contexts[fallback.Name] = fallback;
                this._activeContext = fallback;
                return new ManagedContext { Context = fallback, AutoSave = true, IsTransient = false };
            }

            if (_contexts.TryGetValue(normalized, out var cached))
            {
                this._activeContext = cached;
                return new ManagedContext { Context = cached, AutoSave = false, IsTransient = false };
            }

            var path = ResolveContextFilePath(normalized);
            if (!string.IsNullOrWhiteSpace(path) && File.Exists(path))
            {
                var loaded = LoadContextFromFile(path);
                if (loaded != null)
                {
                    _contexts[loaded.Name] = loaded;
                    this._activeContext = loaded;
                    return new ManagedContext { Context = loaded, AutoSave = true, IsTransient = false };
                }
            }

            var created = new LlamaChatContext
            {
                Name = normalized,
                UpdatedAt = DateTime.UtcNow
            };
            _contexts[created.Name] = created;
            this._activeContext = created;
            return new ManagedContext { Context = created, AutoSave = false, IsTransient = false };
        }

        public async Task<string?> SaveActiveContextAsNewAsync(string? name = null)
        {
            var active = ResolveActiveContext();
            if (active == null)
            {
                return null;
            }

            var finalName = string.IsNullOrWhiteSpace(name) ? Guid.NewGuid().ToString("N") : name;
            var newContext = new LlamaChatContext
            {
                Name = finalName,
                UpdatedAt = DateTime.UtcNow,
                Messages = new List<LlamaChatMessage>(active.Messages)
            };

            var saved = await SaveContextAsync(newContext);
            return saved ? this.LoadedContextFile : null;
        }

        private async Task FinalizeContextAsync(ManagedContext? managed, string prompt, string response)
        {
            if (managed == null)
            {
                return;
            }

            if (!string.IsNullOrWhiteSpace(prompt))
            {
                managed.Context.Messages.Add(new LlamaChatMessage
                {
                    Role = "user",
                    Content = prompt
                });
            }

            if (!string.IsNullOrWhiteSpace(response))
            {
                managed.Context.Messages.Add(new LlamaChatMessage
                {
                    Role = "assistant",
                    Content = response
                });
            }

            managed.Context.UpdatedAt = DateTime.UtcNow;
            this._activeContext = managed.Context;

            if (managed.AutoSave)
            {
                await SaveContextAsync(managed.Context);
            }
        }

        private static string BuildVisionPrompt(LlamaChatContext? context, string prompt, bool includeImageToken)
        {
            var sb = new StringBuilder();
            if (context != null && context.Messages.Count > 0)
            {
                var history = context.Messages
                    .TakeLast(10)
                    .ToList();

                foreach (var message in history)
                {
                    if (string.Equals(message.Role, "assistant", StringComparison.OrdinalIgnoreCase))
                    {
                        sb.Append("<|assistant|>\n").Append(message.Content).Append("<|end|>\n");
                    }
                    else
                    {
                        sb.Append("<|user|>\n").Append(message.Content).Append("<|end|>\n");
                    }
                }
            }

            sb.Append("<|user|>\n");
            if (includeImageToken)
            {
                sb.Append("<|image_1|>\n");
            }

            sb.Append(prompt).Append("<|end|>\n<|assistant|>\n");
            return sb.ToString();
        }

        private static string? GetContextImagePath(string contextName)
        {
            if (string.IsNullOrWhiteSpace(contextName))
            {
                return null;
            }

            var dir = GetContextDirectory();
            Directory.CreateDirectory(dir);
            var safeName = SanitizeContextName(contextName);
            return Path.Combine(dir, safeName + ".image.png");
        }

        public async Task<string> RunPromptAsync(string userPrompt, string? base64Image = null, GenAiSettings? settings = null, string? context = null)
        {
            if (this._genAiModel == null || this._genAiProcessor == null)
            {
                return "Modell nicht geladen.";
            }

            if (string.IsNullOrWhiteSpace(userPrompt))
            {
                return "Prompt ist leer.";
            }

            settings ??= new GenAiSettings();

            var contextName = string.IsNullOrWhiteSpace(context) ? DefaultContextName : context;
            var managedContext = await ResolveContextAsync(contextName);

            return await Task.Run(async () =>
            {
                string tempFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".png");
                string normalizedTemp = tempFile;
                string? imagePath = null;
                try
                {
                    var contextImagePath = managedContext != null
                        ? GetContextImagePath(managedContext.Context.Name)
                        : null;

                    if (!string.IsNullOrWhiteSpace(base64Image))
                    {
                        await File.WriteAllBytesAsync(tempFile, Convert.FromBase64String(base64Image));

                        try
                        {
                            using var img = SixLabors.ImageSharp.Image.Load(tempFile);
                            var norm = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".png");
                            await img.SaveAsPngAsync(norm);
                            try { File.Delete(tempFile); } catch { }
                            normalizedTemp = norm;
                        }
                        catch (Exception ex)
                        {
                            await StaticLogger.LogAsync($"Image normalization failed: {ex.Message}");
                        }

                        if (!string.IsNullOrWhiteSpace(contextImagePath))
                        {
                            File.Copy(normalizedTemp, contextImagePath, true);
                            imagePath = contextImagePath;
                        }
                        else
                        {
                            imagePath = normalizedTemp;
                        }
                    }
                    else if (!string.IsNullOrWhiteSpace(contextImagePath) && File.Exists(contextImagePath))
                    {
                        imagePath = contextImagePath;
                    }

                    if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
                    {
                        await StaticLogger.LogAsync("Kein Bild im Kontext...");
                    }

                    using var images = Images.Load([imagePath]);

                    string fullPrompt = BuildVisionPrompt(managedContext?.Context, userPrompt, includeImageToken: true);

                    using var inputTensors = this._genAiProcessor.ProcessImages(fullPrompt, images);

                    int maxLength = settings.MaxLength;
                    int maxNewTokens = Math.Max(1, settings.MaxNewTokens);

                    GeneratorParams? generatorParams = null;
                    Generator? generator = null;

                    try
                    {
                        generatorParams = new GeneratorParams(this._genAiModel);
                        generatorParams.SetSearchOption("max_length", maxLength);
                        generatorParams.SetSearchOption("do_sample", settings.DoSample);

                        generator = new Generator(this._genAiModel, generatorParams);

                        try
                        {
                            generator.SetInputs(inputTensors);
                        }
                        catch (Exception ex) when (TryGetInputIdsLength(ex.Message, out var inputIdsLength))
                        {
                            int adjustedMaxLength = Math.Max(maxLength, inputIdsLength + maxNewTokens);
                            if (adjustedMaxLength != maxLength)
                            {
                                await StaticLogger.LogAsync($"Adjusting max_length from {maxLength} to {adjustedMaxLength} to fit input size {inputIdsLength}.");
                                maxLength = adjustedMaxLength;
                                generator.Dispose();
                                generatorParams.Dispose();

                                generatorParams = new GeneratorParams(this._genAiModel);
                                generatorParams.SetSearchOption("max_length", maxLength);
                                generatorParams.SetSearchOption("do_sample", settings.DoSample);

                                generator = new Generator(this._genAiModel, generatorParams);
                                generator.SetInputs(inputTensors);
                            }
                            else
                            {
                                throw;
                            }
                        }

                        StringBuilder sb = new StringBuilder();

                        while (!generator.IsDone())
                        {
                            generator.GenerateNextToken();

                            var sequence = generator.GetSequence(0).ToArray();
                            if (sequence.Length > 0)
                            {
                                var lastToken = sequence[^1..];
                                string piece = this._genAiTokenizer!.Decode(lastToken);
                                sb.Append(piece);
                            }
                        }

                        var response = sb.ToString();
                        await FinalizeContextAsync(managedContext, userPrompt, response);
                        return response;
                    }
                    finally
                    {
                        generator?.Dispose();
                        generatorParams?.Dispose();
                    }
                }
                catch (Exception ex)
                {
                    await StaticLogger.LogAsync(ex);
                    return $"Fehler im Loop: {ex.Message}";
                }
                finally
                {
                    try { if (File.Exists(tempFile)) File.Delete(tempFile); } catch { }
                    try
                    {
                        if (!string.IsNullOrEmpty(normalizedTemp)
                            && File.Exists(normalizedTemp)
                            && !string.Equals(normalizedTemp, imagePath, StringComparison.OrdinalIgnoreCase))
                        {
                            File.Delete(normalizedTemp);
                        }
                    }
                    catch { }
                }
            });
        }

        private static string? ResolveContextFilePath(string nameOrRecent)
        {
            var dir = GetContextDirectory();
            Directory.CreateDirectory(dir);

            if (string.Equals(nameOrRecent, RecentContextName, StringComparison.OrdinalIgnoreCase)
                || string.Equals(nameOrRecent, DefaultContextName, StringComparison.OrdinalIgnoreCase))
            {
                var recent = Directory.EnumerateFiles(dir, "*.json")
                    .Select(path => new FileInfo(path))
                    .OrderByDescending(info => info.LastWriteTimeUtc)
                    .FirstOrDefault();

                if (recent != null)
                {
                    return recent.FullName;
                }

                return Path.Combine(dir, SanitizeContextName(DefaultContextName) + ".json");
            }

            var safeName = SanitizeContextName(nameOrRecent);
            return Path.Combine(dir, safeName + ".json");
        }

        private static LlamaChatContext? LoadContextFromFile(string path)
        {
            try
            {
                var json = File.ReadAllText(path, Encoding.UTF8);
                var context = JsonSerializer.Deserialize<LlamaChatContext>(json);
                if (context != null)
                {
                    context.UpdatedAt = File.GetLastWriteTimeUtc(path);
                    return context;
                }
            }
            catch
            {
                // ignore
            }

            return null;
        }

        private static string SanitizeContextName(string name)
        {
            var invalid = Path.GetInvalidFileNameChars();
            var sb = new StringBuilder(name.Length);
            foreach (var ch in name)
            {
                sb.Append(invalid.Contains(ch) ? '_' : ch);
            }
            return sb.ToString();
        }

        private static string GenerateTransientContextName()
        {
            return "transient-" + DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff");
        }

        private static string? NormalizeContextName(string? name)
        {
            if (name == null)
            {
                return null;
            }

            var trimmed = name.Trim();
            if (trimmed.StartsWith("/", StringComparison.Ordinal))
            {
                trimmed = trimmed.TrimStart('/');
            }

            return trimmed;
        }

        public static string GetContextDirectory()
        {
            var dir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "OnnxPeriment_Contexts");
            return dir;
        }

        // I/O
    }
}
