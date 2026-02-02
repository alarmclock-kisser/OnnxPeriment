using LocalLlmTestDataGenerator.Core;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Runtime.Versioning;
using System.Management;

namespace OnnxPeriment.Core
{
    [SupportedOSPlatform("windows")]
    public class CpuMonitor
    {
        private bool _enabled = true;
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

        private Timer? StatisticsTimer;
        private List<PerformanceCounter>? _counters;
        private bool _countersWarmed;
        private readonly SemaphoreSlim _usageLock = new(1, 1);

        public event EventHandler<float[]>? UsageUpdated;

        public CpuMonitor(int intervallMs = 250)
        {
            this.IntervalMs = Math.Max(1, intervallMs);
            this.StatisticsTimer = new Timer(_ => _ = this.PollUsagesAsync(), null, Timeout.Infinite, Timeout.Infinite);

            if (this.Enabled)
            {
                this.StartTimer();
            }
        }

        [SupportedOSPlatform("windows")]
        public async Task<float[]?> GetCpuUsagesAsync()
        {
            await this._usageLock.WaitAsync();
            try
            {
                int cpuCount = Environment.ProcessorCount;
                float[] usages = new float[cpuCount + 1];

                try
                {
                    EnsureCounters(cpuCount);
                    if (this._counters == null || this._counters.Count == 0)
                    {
                        return usages;
                    }

                    if (!this._countersWarmed)
                    {
                        foreach (var counter in this._counters)
                        {
                            _ = counter.NextValue();
                        }
                        this._countersWarmed = true;
                        await Task.Delay(Math.Min(250, this.IntervalMs));
                    }

                    usages[0] = this._counters[0].NextValue();
                    for (int i = 0; i < cpuCount; i++)
                    {
                        usages[i + 1] = this._counters[i + 1].NextValue();
                    }
                }
                catch (Exception ex)
                {
                    await StaticLogger.LogAsync($"Error retrieving CPU usages: {ex.Message}");
                }

                return usages;
            }
            finally
            {
                this._usageLock.Release();
            }
        }

        [SupportedOSPlatform("windows")]
        public async Task<(double Used, double Total)> GetRamUsageMbAsync()
        {
            try
            {
                var ramCounter = new PerformanceCounter("Memory", "Available MBytes");
                float available = ramCounter.NextValue();
                float total = GetTotalPhysicalMemory();
                float used = total - available;

                return (used, total);
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync($"Error retrieving RAM usage: {ex.Message}");
                return (0, 0);
            }
        }

        [SupportedOSPlatform("windows")]
        public float GetTotalPhysicalMemory()
        {
            try
            {
                // Alternative Methode ohne Microsoft.VisualBasic.Devices.ComputerInfo
                var ci = new System.Management.ManagementObjectSearcher("SELECT TotalPhysicalMemory FROM Win32_ComputerSystem");
                foreach (var o in ci.Get())
                {
                    if (o["TotalPhysicalMemory"] is ulong totalMemory)
                    {
                        return totalMemory / (1024 * 1024);
                    }
                }
                return 0;
            }
            catch
            {
                return 0;
            }
        }


        [SupportedOSPlatform("windows")]
        public async Task<ImageObj?> GenerateUsageGraphicAsync(int width = 200, int height = 100, float[]? cpuUsages = null, string foreColor = "#32CD32", string backColor = "#2F4F4F", bool indicateThreadIds = true)
        {
            cpuUsages ??= await this.GetCpuUsagesAsync();

            try
            {
                ImageObj? obj = await Task.Run(() =>
                {
                    return new ImageObj(width, height, cpuUsages ?? [], foreColor, backColor, indicateThreadIds);
                });

                return obj;
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync($"Error generating CPU usage graphic: {ex.Message}");
                return null;
            }
        }

        public async Task SetIntervalAsync(int intervalMs)
        {
            this.IntervalMs = Math.Max(1, intervalMs);
            if (this.Enabled)
            {
                this.StatisticsTimer?.Change(0, this.IntervalMs);
            }
            await Task.CompletedTask;
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
            this.StatisticsTimer?.Change(0, this.IntervalMs);
        }

        private void StopTimer()
        {
            this.StatisticsTimer?.Change(Timeout.Infinite, Timeout.Infinite);
        }

        private async Task PollUsagesAsync()
        {
            if (!this.Enabled)
            {
                return;
            }

            var usages = await this.GetCpuUsagesAsync();
            if (usages != null)
            {
                this.UsageUpdated?.Invoke(this, usages);
            }
        }

        private void EnsureCounters(int cpuCount)
        {
            if (this._counters != null)
            {
                return;
            }

            var counters = new List<PerformanceCounter>(cpuCount + 1)
            {
                new("Processor", "% Processor Time", "_Total", true)
            };

            for (int i = 0; i < cpuCount; i++)
            {
                counters.Add(new PerformanceCounter("Processor", "% Processor Time", i.ToString(), true));
            }

            this._counters = counters;
        }
    }
}
