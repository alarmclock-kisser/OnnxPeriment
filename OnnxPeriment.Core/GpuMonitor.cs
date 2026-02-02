using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.Versioning;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxPeriment.Core
{
    public sealed record GpuUsage(float LoadPercent, double VramUsedMb, double VramTotalMb);

    [SupportedOSPlatform("windows")]
    public class GpuMonitor
    {
        private bool _enabled = true;
        private readonly SemaphoreSlim _usageLock = new(1, 1);
        private Timer? _statisticsTimer;
        private List<PerformanceCounter>? _engineCounters;
        private PerformanceCounter? _dedicatedUsage;
        private PerformanceCounter? _dedicatedLimit;
        private PerformanceCounter? _sharedUsage;
        private PerformanceCounter? _sharedLimit;
        private bool _countersWarmed;
        private string? _selectedKey;
        private List<string>? _gpuKeys;

        public bool Enabled
        {
            get => this._enabled;
            set
            {
                if (this._enabled == value)
                {
                    return;
                }

                this._enabled = value;
                if (this._enabled)
                {
                    this.StartTimer();
                }
                else
                {
                    this.StopTimer();
                }
            }
        }

        public int IntervalMs { get; private set; }

        public int GpuIndex { get; private set; }

        public IReadOnlyList<string> AvailableGpuKeys => this._gpuKeys ?? [];

        public event EventHandler<GpuUsage>? UsageUpdated;

        [SupportedOSPlatform("windows")]
        public GpuMonitor(int intervalMs = 250, int gpuIndex = 0)
        {
            this.IntervalMs = Math.Max(1, intervalMs);
            this.GpuIndex = Math.Max(0, gpuIndex);
            this._statisticsTimer = new Timer(_ => _ = this.PollUsageAsync(), null, Timeout.Infinite, Timeout.Infinite);

            if (this.Enabled)
            {
                this.StartTimer();
            }
        }

        [SupportedOSPlatform("windows")]
        public async Task<GpuUsage?> GetUsageAsync()
        {
            await this._usageLock.WaitAsync();
            try
            {
                EnsureCounters();
                if (this._engineCounters == null || this._engineCounters.Count == 0)
                {
                    return null;
                }

                if (!this._countersWarmed)
                {
                    foreach (var counter in this._engineCounters)
                    {
                        _ = counter.NextValue();
                    }
                    _ = this._dedicatedUsage?.NextValue();
                    _ = this._dedicatedLimit?.NextValue();
                    _ = this._sharedUsage?.NextValue();
                    _ = this._sharedLimit?.NextValue();
                    this._countersWarmed = true;
                    await Task.Delay(Math.Min(250, this.IntervalMs));
                }

                float load = 0f;
                foreach (var counter in this._engineCounters)
                {
                    load += counter.NextValue();
                }

                load = Math.Clamp(load, 0f, 100f);

                double dedicatedUsed = this._dedicatedUsage?.NextValue() ?? 0;
                double dedicatedTotal = this._dedicatedLimit?.NextValue() ?? 0;
                double sharedUsed = this._sharedUsage?.NextValue() ?? 0;
                double sharedTotal = this._sharedLimit?.NextValue() ?? 0;

                double used = dedicatedUsed > 0 ? dedicatedUsed : sharedUsed;
                double total = dedicatedTotal > 0 ? dedicatedTotal : sharedTotal;

                double usedMb = used / (1024 * 1024);
                double totalMb = total / (1024 * 1024);

                return new GpuUsage(load, usedMb, totalMb);
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync($"Error retrieving GPU usage: {ex.Message}");
                return null;
            }
            finally
            {
                this._usageLock.Release();
            }
        }

        public async Task SetIntervalAsync(int intervalMs)
        {
            this.IntervalMs = Math.Max(1, intervalMs);
            if (this.Enabled)
            {
                this._statisticsTimer?.Change(0, this.IntervalMs);
            }
            await Task.CompletedTask;
        }

        public void SetGpuIndex(int gpuIndex)
        {
            this.GpuIndex = Math.Max(0, gpuIndex);
            this.ResetCounters();
        }

        public void EnableMonitoring()
        {
            this.Enabled = true;
        }

        public void DisableMonitoring()
        {
            this.Enabled = false;
        }

        private void StartTimer()
        {
            this._statisticsTimer?.Change(0, this.IntervalMs);
        }

        private void StopTimer()
        {
            this._statisticsTimer?.Change(Timeout.Infinite, Timeout.Infinite);
        }

        [SupportedOSPlatform("windows")]
        private async Task PollUsageAsync()
        {
            if (!this.Enabled)
            {
                return;
            }

            var usage = await this.GetUsageAsync();
            if (usage != null)
            {
                this.UsageUpdated?.Invoke(this, usage);
            }
        }

        [SupportedOSPlatform("windows")]
        private void EnsureCounters()
        {
            if (this._engineCounters != null)
            {
                return;
            }

            try
            {
                var engineCategory = new PerformanceCounterCategory("GPU Engine");
                var engineInstances = engineCategory.GetInstanceNames();
                this._gpuKeys = [.. engineInstances
                    .Select(ParseGpuKey)
                    .Where(key => !string.IsNullOrWhiteSpace(key))
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .OrderBy(key => key)];

                if (this._gpuKeys.Count == 0)
                {
                    this._engineCounters = new List<PerformanceCounter>();
                    return;
                }

                this._selectedKey = this._gpuKeys[Math.Min(this.GpuIndex, this._gpuKeys.Count - 1)];

                var engineFilter = new[] { "engtype_3D", "engtype_CUDA", "engtype_Compute" };
                this._engineCounters = engineInstances
                    .Where(name => name.Contains(this._selectedKey, StringComparison.OrdinalIgnoreCase))
                    .Where(name => engineFilter.Any(filter => name.Contains(filter, StringComparison.OrdinalIgnoreCase)))
                    .Select(name => new PerformanceCounter("GPU Engine", "Utilization Percentage", name, true))
                    .ToList();

                var memoryCategory = new PerformanceCounterCategory("GPU Adapter Memory");
                var memoryInstances = memoryCategory.GetInstanceNames();
                var memoryInstance = memoryInstances.FirstOrDefault(name => name.Contains(this._selectedKey, StringComparison.OrdinalIgnoreCase))
                    ?? memoryInstances.FirstOrDefault();

                if (!string.IsNullOrWhiteSpace(memoryInstance))
                {
                    this._dedicatedUsage = new PerformanceCounter("GPU Adapter Memory", "Dedicated Usage", memoryInstance, true);
                    this._dedicatedLimit = new PerformanceCounter("GPU Adapter Memory", "Dedicated Limit", memoryInstance, true);
                    this._sharedUsage = new PerformanceCounter("GPU Adapter Memory", "Shared Usage", memoryInstance, true);
                    this._sharedLimit = new PerformanceCounter("GPU Adapter Memory", "Shared Limit", memoryInstance, true);
                }
            }
            catch (Exception ex)
            {
                StaticLogger.Log($"Failed to initialize GPU counters: {ex.Message}");
                this._engineCounters = new List<PerformanceCounter>();
            }
        }

        private void ResetCounters()
        {
            this._engineCounters = null;
            this._dedicatedUsage = null;
            this._dedicatedLimit = null;
            this._sharedUsage = null;
            this._sharedLimit = null;
            this._countersWarmed = false;
            this._selectedKey = null;
        }

        private static string? ParseGpuKey(string instanceName)
        {
            if (string.IsNullOrWhiteSpace(instanceName))
            {
                return null;
            }

            var physIndex = instanceName.IndexOf("phys_", StringComparison.OrdinalIgnoreCase);
            if (physIndex >= 0)
            {
                var start = physIndex;
                var end = instanceName.IndexOf('_', start + 5);
                return end > start ? instanceName.Substring(start, end - start) : instanceName.Substring(start);
            }

            var luidIndex = instanceName.IndexOf("luid_", StringComparison.OrdinalIgnoreCase);
            if (luidIndex >= 0)
            {
                var start = luidIndex;
                var end = instanceName.IndexOf('_', start + 5);
                return end > start ? instanceName.Substring(start, end - start) : instanceName.Substring(start);
            }

            return null;
        }
    }
}
