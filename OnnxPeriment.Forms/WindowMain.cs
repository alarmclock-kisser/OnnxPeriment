using LocalLlmTestDataGenerator.Core;
using OnnxPeriment.Core;
using OnnxPeriment.Runtime;
using Timer = System.Windows.Forms.Timer;
using System.Threading;
using System.Text;
using OnnxPeriment.Shared;
using System.Threading.Tasks;

namespace OnnxPeriment.Forms
{
    public partial class WindowMain : Form
    {
        internal readonly OnnxService Onnx;
        internal readonly LlamaService Llama;
        internal readonly ImageCollection Images;



        private Timer? ResponseTimer = null;
        internal DateTime? PromptSent = null;



        internal bool ZoomImages => this.checkBox_zoomImage.Checked;
        internal bool AddImage => this.checkBox_includeImage.Checked;


        public WindowMain()
        {
            InitializeComponent();
            // Ensure logger posts binding updates on the UI thread
            StaticLogger.SetUiContext(SynchronizationContext.Current!);

            this.Bind_ListBox_Log(this.listBox_log, true);

            this.Onnx = new OnnxService();
            this.Llama = new LlamaService();
            this.Images = new ImageCollection(true);
            this.Update_Numeric_Images();

            this.Load += (_, __) => UpdateBackendButtons();
        }

        private void UpdateBackendButtons()
        {
            try
            {
                var onnxBackends = this.Onnx.VerifyBackends(testLoad: false);
                var llamaBackends = this.Llama.VerifyBackends(testLoad: false);

                bool onnxMissing = onnxBackends.Any(b => !b.AssemblyAvailable || !b.NativeLibraryFound);
                bool llamaMissing = llamaBackends.Any(b => !b.AssemblyAvailable || !b.NativeLibraryFound);

                this.button_downloadIOnnxCuda.Visible = onnxMissing;
                this.button_downloadIOnnxCuda.Enabled = onnxMissing;

                this.button_downloadLlamaCuda.Visible = llamaMissing;
                this.button_downloadLlamaCuda.Enabled = llamaMissing;
            }
            catch (Exception ex)
            {
                StaticLogger.Log($"Failed to verify backend dependencies: {ex.Message}");
            }
        }



        // Functions
        private void Bind_ListBox_Log(ListBox listBox, bool autoScroll = false)
        {
            listBox.SuspendLayout();
            listBox.Items.Clear();

            listBox.DataSource = StaticLogger.LogEntriesBindingList;

            if (autoScroll)
            {
                StaticLogger.LogEntriesBindingList.ListChanged += (s, e) =>
                {
                    if (e.ListChangedType == System.ComponentModel.ListChangedType.ItemAdded)
                    {
                        if (listBox.InvokeRequired)
                        {
                            listBox.BeginInvoke((MethodInvoker) (() =>
                                listBox.TopIndex = listBox.Items.Count - 1));
                        }
                        else
                        {
                            listBox.TopIndex = listBox.Items.Count - 1;
                        }
                    }
                };
            }

            listBox.ResumeLayout();
        }

        private void Update_Numeric_Images()
        {
            if (this.Images.Count == 0)
            {
                this.numericUpDown_images.Value = 0;
            }
            else if (this.numericUpDown_images.Value > this.Images.Count)
            {
                this.numericUpDown_images.Value = this.Images.Count;
            }

            this.numericUpDown_images.Maximum = this.Images.Count;
        }

        internal async Task ShowImageAsync(int? index = null)
        {
            int id = index ?? (int) this.numericUpDown_images.Value - 1;

            if (this.numericUpDown_images.Value > 0)
            {
                var imageObj = this.Images[id];
                if (imageObj == null)
                {
                    this.pictureBox_view.Image?.Dispose();
                    this.pictureBox_view.Image = null;
                    return;
                }

                var pixelData = await imageObj.GetImageDataAsync(false);
                int width = imageObj.Width;
                int height = imageObj.Height;

                // Konvertiere RGBA -> BGRA falls nötig
                var bgra = new byte[pixelData.Length];
                for (int i = 0; i < width * height; i++)
                {
                    int src = i * 4;
                    bgra[src + 0] = pixelData[src + 2]; // B <- R
                    bgra[src + 1] = pixelData[src + 1]; // G
                    bgra[src + 2] = pixelData[src + 0]; // R <- B
                    bgra[src + 3] = pixelData[src + 3]; // A
                }

                var bmp = new System.Drawing.Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                var rect = new System.Drawing.Rectangle(0, 0, width, height);
                var bmpData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.WriteOnly, bmp.PixelFormat);
                System.Runtime.InteropServices.Marshal.Copy(bgra, 0, bmpData.Scan0, bgra.Length);
                bmp.UnlockBits(bmpData);

                this.pictureBox_view.Image?.Dispose();
                this.pictureBox_view.Image = bmp;
                if (this.ZoomImages)
                {
                    this.pictureBox_view.SizeMode = PictureBoxSizeMode.Zoom;
                }
                else
                {
                    this.pictureBox_view.SizeMode = PictureBoxSizeMode.CenterImage;
                }
            }
            else
            {
                this.pictureBox_view.Image?.Dispose();
                this.pictureBox_view.Image = null;
            }
        }

        internal void UpdateStatus()
        {
            // Check if invoked from another thread
            if (this.label_status.InvokeRequired)
            {
                this.label_status.BeginInvoke((MethodInvoker) (() =>
                {
                    this.UpdateStatus();
                }));
                return;
            }

            // Check Onnx & Llama model loaded status
            string? onnxLoaded = this.Onnx.LoadedModelPath;
            string? llamaLoaded = this.Llama.LoadedModelPath;

            if (string.IsNullOrEmpty(onnxLoaded) && string.IsNullOrEmpty(llamaLoaded))
            {
                this.label_status.Text = "No model loaded.";
                this.label_status.ForeColor = System.Drawing.Color.DarkGray;
            }
            else if (!string.IsNullOrEmpty(onnxLoaded) && !string.IsNullOrEmpty(llamaLoaded))
            {
                this.label_status.Text = $"ONNX Model: {Path.GetFileName(onnxLoaded)} | Llama Model: {Path.GetFileName(llamaLoaded)}";
                this.label_status.ForeColor = System.Drawing.Color.Green;
            }
            else if (!string.IsNullOrEmpty(onnxLoaded))
            {
                this.label_status.Text = $"ONNX Model loaded: {Path.GetFileName(onnxLoaded)}";
                this.label_status.ForeColor = System.Drawing.Color.Green;

                this.label_context.Text = "ONNX CTX";
            }
            else if (!string.IsNullOrEmpty(llamaLoaded))
            {
                this.label_status.Text = $"Llama Model loaded: {Path.GetFileName(llamaLoaded)}";
                this.label_status.ForeColor = System.Drawing.Color.Green;

                if (string.IsNullOrEmpty(this.Llama.LoadedContextFile))
                {
                    this.label_context.Text = "Temporary Context - not saved yet!";
                }
                else
                {
                    this.label_context.Text = Path.GetFileNameWithoutExtension(this.Llama.LoadedContextFile);
                }
            }
        }



        // UI
        private async void numericUpDown_images_ValueChanged(object sender, EventArgs e)
        {
            await this.ShowImageAsync();

            if (this.numericUpDown_images.Value > 0)
            {
                var img = this.Images[(int) this.numericUpDown_images.Value - 1];
                if (img == null)
                {
                    if (this.label_imageInfo.InvokeRequired)
                    {
                        this.label_imageInfo.BeginInvoke((MethodInvoker) (() =>
                        {
                            this.label_imageInfo.Text = $"Image not found.";
                        }));
                    }
                    else
                    {
                        this.label_imageInfo.Text = $"Image not found.";
                    }
                    return;
                }

                if (this.label_imageInfo.InvokeRequired)
                {
                    this.label_imageInfo.BeginInvoke((MethodInvoker) (() =>
                    {
                        this.label_imageInfo.Text = $"{img.Id} - {img.SizeInKb.ToString("F2")} KB ({img.Width}x{img.Height} px)";
                    }));
                }
                else
                {
                    this.label_imageInfo.Text = $"{img.Id} - {img.SizeInKb.ToString("F2")} KB ({img.Width}x{img.Height} px)";
                }

                if (this.button_deleteImage.InvokeRequired)
                {
                    this.button_deleteImage.BeginInvoke((MethodInvoker) (() =>
                    {
                        this.button_deleteImage.Enabled = true;
                    }));
                }
                else
                {
                    this.button_deleteImage.Enabled = true;
                }

                this.checkBox_includeImage.Invoke((MethodInvoker) (() =>
                {
                    this.checkBox_includeImage.Checked = true;
                }));
            }
            else
            {
                if (this.label_imageInfo.InvokeRequired)
                {
                    this.label_imageInfo.BeginInvoke((MethodInvoker) (() =>
                    {
                        this.label_imageInfo.Text = $"No image selected.";
                    }));
                }
                else
                {
                    this.label_imageInfo.Text = $"No image selected.";
                }

                if (this.button_deleteImage.InvokeRequired)
                {
                    this.button_deleteImage.BeginInvoke((MethodInvoker) (() =>
                    {
                        this.button_deleteImage.Enabled = false;
                    }));
                }
                else
                {
                    this.button_deleteImage.Enabled = false;
                }

                this.checkBox_includeImage.Invoke((MethodInvoker) (() =>
                {
                    this.checkBox_includeImage.Checked = false;
                }));
            }
        }

        private void checkBox_centerImage_CheckedChanged(object sender, EventArgs e)
        {
            if (this.checkBox_zoomImage.Checked)
            {
                this.pictureBox_view.SizeMode = PictureBoxSizeMode.Zoom;
            }
            else
            {
                this.pictureBox_view.SizeMode = PictureBoxSizeMode.CenterImage;
            }
        }



        // Images
        private async void button_importImage_Click(object sender, EventArgs e)
        {
            string browseDir = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
            using OpenFileDialog ofd = new()
            {
                InitialDirectory = browseDir,
                Filter = "Image Files (*.png;*.jpg;*.jpeg;*.bmp;*.gif)|*.png;*.jpg;*.jpeg;*.bmp;*.gif|All Files (*.*)|*.*",
                Title = "Select Image Files",
                Multiselect = true
            };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                var importTasks = ofd.FileNames.Select(filePath => this.Images.ImportImageAsync(filePath));

                await Task.WhenAll(importTasks);

                this.Update_Numeric_Images();
                StaticLogger.Log($"Imported {ofd.FileNames.Length} image(s).");
                this.numericUpDown_images.Invoke((MethodInvoker) (() =>
                {
                    this.numericUpDown_images.Value = this.Images.Count;
                }));
            }
        }

        private void button_deleteImage_Click(object sender, EventArgs e)
        {
            if (this.numericUpDown_images.Value <= 0)
            {
                return;
            }

            var imageObj = this.Images[(int) this.numericUpDown_images.Value - 1];
            if (imageObj != null)
            {
                this.Images.RemoveImage(imageObj.Id, true);
                this.Update_Numeric_Images();
            }
        }




        // Load models + ctx
        private async void button_loadOnnxModel_Click(object sender, EventArgs e)
        {
            string browseDir = this.Onnx.ModelDirectories.FirstOrDefault() ?? Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            using OpenFileDialog ofd = new()
            {
                InitialDirectory = browseDir,
                Filter = "ONNX Model Files (*.onnx)|*.onnx|All Files (*.*)|*.*",
                Title = "Select an ONNX Model File"
            };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                this.BeginInvoke((MethodInvoker) (() =>
                {
                    this.button_loadOnnxModel.Enabled = false;
                    this.button_loadLlamaModel.Enabled = false;
                }));
                string modelPath = ofd.FileName;
                string? loadResult = await this.Onnx.LoadGenAiModelAsync(modelPath, this.checkBox_enableCuda.Checked);
                if (loadResult != null)
                {
                    MessageBox.Show($"Failed to load model: {loadResult}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    this.BeginInvoke((MethodInvoker) (() =>
                    {
                        this.button_loadOnnxModel.Enabled = true;
                        this.button_loadLlamaModel.Enabled = true;
                    }));
                }
                else
                {
                    StaticLogger.Log($"Model loaded: {modelPath}");
                    this.checkBox_enableCuda.Invoke((MethodInvoker) (() =>
                    {
                        this.checkBox_enableCuda.Enabled = false;
                    }));
                }
            }

            this.UpdateStatus();
        }

        private async void button_loadLlamaModel_Click(object sender, EventArgs e)
        {
            string browseDir = this.Llama.ModelDirectories.FirstOrDefault() ?? Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            using OpenFileDialog ofd = new()
            {
                InitialDirectory = browseDir,
                Filter = "Llama Model Files (*.bin;*.gguf)|*.bin;*.gguf|All Files (*.*)|*.*",
                Title = "Select a Llama Model File"
            };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                this.BeginInvoke((MethodInvoker) (() =>
                {
                    this.button_loadOnnxModel.Enabled = false;
                    this.button_loadLlamaModel.Enabled = false;
                }));

                string modelPath = ofd.FileName;
                LlamaModelLoadOptions options = new LlamaModelLoadOptions
                {
                    UseCuda = this.checkBox_enableCuda.Checked,
                };
                string? loadResult = await this.Llama.LoadModelAsync(modelPath, options);
                if (string.IsNullOrEmpty(loadResult))
                {
                    MessageBox.Show($"Failed to load model: {modelPath}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    this.BeginInvoke((MethodInvoker) (() =>
                    {
                        this.button_loadOnnxModel.Enabled = true;
                        this.button_loadLlamaModel.Enabled = true;
                    }));
                }
                else
                {
                    StaticLogger.Log($"Model loaded: {loadResult}");
                    this.checkBox_enableCuda.Invoke((MethodInvoker) (() =>
                    {
                        this.checkBox_enableCuda.Enabled = false;
                    }));
                }
            }

            this.UpdateStatus();
        }

        private async void button_loadContext_Click(object sender, EventArgs e)
        {
            string contextsDir = Path.Combine(Path.GetFullPath(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData)), "OnnxPeriment_Contexts");
            OpenFileDialog ofd = new()
            {
                InitialDirectory = contextsDir,
                Filter = "JSON Context Files (*.json)|*.json|All Files (*.*)|*.*",
                Title = "Select a Context File (.json)",
                Multiselect = false
            };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                string contextPath = ofd.FileName;
                try
                {
                    if (this.Onnx.ModelIsLoaded)
                    {

                    }
                    else if (this.Llama.ModelIsLoaded)
                    {
                        var ctx = await this.Llama.LoadContextAsync(contextPath);
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to load context: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void numericUpDown_messages_ValueChanged(object sender, EventArgs e)
        {
            // Max to current contexts message-pairs (uestion + answer)
            if (this.Onnx.ModelIsLoaded)
            {

            }
            else if (this.Llama.ModelIsLoaded)
            {

            }
        }

        private void numericUpDown_messages_Click(object sender, EventArgs e)
        {
            // If right clicked jump to empty (new message, latest + 1)
            if (this.Onnx.ModelIsLoaded)
            {

            }
            else if (this.Llama.ModelIsLoaded)
            {

            }
        }



        // Request / Response
        private async void textBox_prompt_KeyDown(object sender, KeyEventArgs e)
        {
            // Check if empty (-> Disable send button)
            if (string.IsNullOrWhiteSpace(this.textBox_prompt.Text))
            {
                this.button_send.BeginInvoke((MethodInvoker) (() =>
                {
                    this.button_send.Enabled = false;
                }));
            }
            else
            {
                this.button_send.BeginInvoke((MethodInvoker) (() =>
                {
                    this.button_send.Enabled = true;
                }));
            }

            // Check for Enter key + Shift to add a line break
            if (e.KeyCode == Keys.Enter && e.Shift)
            {
                // Add line break
                int selectionStart = this.textBox_prompt.SelectionStart;
                this.textBox_prompt.Invoke((MethodInvoker) (() =>
                {
                    this.textBox_prompt.Text = this.textBox_prompt.Text.Insert(selectionStart, Environment.NewLine);
                    this.textBox_prompt.SelectionStart = selectionStart + Environment.NewLine.Length;
                }));
            }

            // Check if only Enter is hit -> Send request
            if (e.KeyCode == Keys.Enter && !e.Shift)
            {
                e.SuppressKeyPress = true; // Verhindert den Zeilenumbruch
                this.button_send.Invoke((MethodInvoker) (() =>
                {
                    this.button_send.PerformClick();
                }));
            }
        }

        private void listBox_log_DoubleClick(object sender, EventArgs e)
        {
            if (ModifierKeys.HasFlag(Keys.Control))
            {
                // Get all log entries
                StringBuilder allLogs = new StringBuilder();
                foreach (var entry in StaticLogger.LogEntriesBindingList)
                {
                    allLogs.AppendLine(entry);
                }
                Clipboard.SetText(allLogs.ToString());
                MessageBox.Show("All log entries have been copied to the clipboard.", "Logs Copied", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            if (this.listBox_log.SelectedItem != null)
            {
                string logEntry = this.listBox_log.SelectedItem.ToString() ?? string.Empty;
                Clipboard.SetText(logEntry);
            }
        }

        private async void button_send_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrWhiteSpace(this.textBox_prompt.Text))
            {
                return;
            }
            string prompt = this.textBox_prompt.Text.Trim();
            StaticLogger.Log($"User Prompt: {prompt}");
            this.textBox_prompt.Clear();

            string? image = null;
            if (this.AddImage && this.numericUpDown_images.Value > 0)
            {
                var imgObj = this.Images[(int) this.numericUpDown_images.Value - 1];
                if (imgObj != null)
                {
                    try
                    {
                        var tempExport = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".png");
                        var exportPath = await imgObj.ExportAsync(tempExport, "png");
                        if (!string.IsNullOrWhiteSpace(exportPath) && File.Exists(exportPath))
                        {
                            var bytes = await File.ReadAllBytesAsync(exportPath);
                            image = Convert.ToBase64String(bytes);
                            File.Delete(exportPath);
                            StaticLogger.Log($"Prepared OCR image: {bytes.Length} bytes, base64 length {image.Length}.");
                        }
                        else
                        {
                            StaticLogger.Log("Failed to export image to PNG before OCR request.");
                        }
                    }
                    catch (Exception ex)
                    {
                        StaticLogger.Log($"Error exporting image for OCR: {ex.Message}");
                    }
                }
            }

            if (this.AddImage && string.IsNullOrWhiteSpace(image))
            {
                StaticLogger.Log("OCR request cancelled: no valid image data prepared.");
                return;
            }

            this.PromptSent = DateTime.UtcNow;
            this.ResponseTimer?.Stop();
            this.ResponseTimer = new Timer
            {
                Interval = 100
            };
            this.ResponseTimer.Tick += (s, args) =>
            {
                this.label_timer.Invoke((MethodInvoker) (() =>
                {
                    this.label_timer.Text = $"Waiting for response... {(DateTime.UtcNow - this.PromptSent.Value).TotalSeconds:F1}s";
                }));
            };
            this.ResponseTimer.Start();

            this.BeginInvoke((MethodInvoker) (() =>
            {
                this.button_send.Enabled = false;
                this.textBox_prompt.Enabled = false;
            }));

            string? response = null;
            if (this.Llama.ModelIsLoaded)
            {
                if (string.IsNullOrEmpty(image))
                {
                    response = await this.Llama.GenerateTextToTextAsync(prompt);
                }
                else
                {
                    response = await this.Llama.GenerateTextWithImagesToTextAsync(prompt, [image]);
                }

                var stats = this.Llama.LastStats;
                this.label_stats.BeginInvoke((MethodInvoker) (() =>
                {
                    this.label_stats.Text = stats == null
                        ? "No stats available."
                        : $"tokens={stats.TokenCount}, elapsed={stats.ElapsedSeconds:F2}s, tokens/s={stats.TokensPerSecond:F2}";
                }));
            }
            else if (this.Onnx.ModelIsLoaded)
            {
                response = await this.Onnx.RunOcrPromptAsync(image ?? string.Empty, prompt);
                this.label_stats.BeginInvoke((MethodInvoker) (() =>
                {
                    this.label_stats.Text = "No stats available.";
                }));
            }

            if (response != null)
            {
                this.label_timer.Invoke((MethodInvoker) (() =>
                {
                    this.label_timer.Text = $"Response received in {(DateTime.UtcNow - this.PromptSent.Value).TotalSeconds:F1}s";
                }));
                StaticLogger.Log($"Model Response: {response}");
            }

            this.label_timer.Invoke((MethodInvoker) (() =>
            {
                this.label_timer.Text = $"Idle";
            }));

            this.textBox_response.Invoke((MethodInvoker) (() =>
            {
                this.textBox_response.AppendText(response + Environment.NewLine + Environment.NewLine);
                this.textBox_response.SelectionStart = this.textBox_response.Text.Length;
                this.textBox_response.ScrollToCaret();
            }));

            this.ResponseTimer?.Stop();
            this.BeginInvoke((MethodInvoker) (() =>
            {
                this.button_send.Enabled = true;
                this.textBox_prompt.Enabled = true;
            }));

            await StaticLogger.LogAsync("Elapsed time for prompt: " + (DateTime.UtcNow - this.PromptSent.Value).TotalSeconds.ToString("F3") + " sec.");
        }

        private void button_backendsOnnx_Click(object sender, EventArgs e)
        {
            var result = this.Onnx.VerifyBackends();

            StringBuilder sb = new StringBuilder();
            sb.AppendLine("ONNX Backend Verification Results:");
            foreach (var backend in result)
            {
                sb.AppendLine(backend.ToString(true) + Environment.NewLine);
            }

            sb.AppendLine();
            sb.AppendLine("Press OK to copy to clipboard.");

            var dialog = MessageBox.Show(sb.ToString(), "ONNX Backends", MessageBoxButtons.OKCancel, MessageBoxIcon.Information);
            if (dialog == DialogResult.OK)
            {
                var copyText = new StringBuilder();
                copyText.AppendLine("ONNX Backend Verification Results:");
                foreach (var backend in result)
                {
                    copyText.AppendLine(backend.ToString(false) + Environment.NewLine);
                }
                Clipboard.SetText(copyText.ToString());
            }
        }

        private void button_backendsLlama_Click(object sender, EventArgs e)
        {
            List<LlamaBackendStatus> result;
            try
            {
                result = this.Llama.VerifyBackends(testLoad: false);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to verify Llama backends: {ex.Message}", "Llama Backends", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            StringBuilder sb = new StringBuilder();
            sb.AppendLine("Llama Backend Verification Results:");
            foreach (var backend in result)
            {
                sb.AppendLine(backend.ToString(true) + Environment.NewLine);
            }

            sb.AppendLine();
            sb.AppendLine("Press OK to copy to clipboard.");

            var dlg = MessageBox.Show(sb.ToString(), "Llama Backends", MessageBoxButtons.OKCancel, MessageBoxIcon.Information);
            if (dlg == DialogResult.OK)
            {
                var copyText = new StringBuilder();
                copyText.AppendLine("Llama Backend Verification Results:");
                foreach (var backend in result)
                {
                    copyText.AppendLine(backend.ToString(false) + Environment.NewLine);
                }
                Clipboard.SetText(copyText.ToString());
            }
        }





        // DLL downloads
        private async void button_downloadIOnnxCuda_Click(object sender, EventArgs e)
        {
            var result = await this.Onnx.EnsureCudaDependenciesAsync();
            this.BeginInvoke((MethodInvoker) (() =>
            {
                this.button_backendsLlama.Enabled = false;
                this.button_backendsOnnx.Enabled = false;
                this.button_downloadIOnnxCuda.Enabled = false;
                this.button_downloadLlamaCuda.Enabled = false;
                this.button_loadLlamaModel.Enabled = false;
                this.button_loadOnnxModel.Enabled = false;
            }));

            if (result?.Count > 0)
            {
                MessageBox.Show($"{result.Count} ONNX CUDA dependencies are installed.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                this.BeginInvoke((MethodInvoker) (() =>
                {
                    this.button_downloadLlamaCuda.Enabled = true;
                    this.button_backendsOnnx.Enabled = true;
                    this.button_loadLlamaModel.Enabled = true;
                    this.button_loadOnnxModel.Enabled = true;
                    this.button_backendsLlama.Enabled = true;
                }));
            }
            else
            {
                MessageBox.Show("Failed to install ONNX CUDA dependencies. Please check the logs for details.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                this.BeginInvoke((MethodInvoker) (() =>
                {
                    this.button_backendsLlama.Enabled = true;
                    this.button_backendsOnnx.Enabled = true;
                    this.button_downloadIOnnxCuda.Enabled = true;
                    this.button_downloadLlamaCuda.Enabled = true;
                    this.button_loadLlamaModel.Enabled = true;
                    this.button_loadOnnxModel.Enabled = true;
                }));
            }
        }

        private async void button_downloadLlamaCuda_Click(object sender, EventArgs e)
        {
            var result = await this.Llama.EnsureCudaDependenciesAsync();
            this.BeginInvoke((MethodInvoker) (() =>
            {
                this.button_backendsLlama.Enabled = false;
                this.button_backendsOnnx.Enabled = false;
                this.button_downloadIOnnxCuda.Enabled = false;
                this.button_downloadLlamaCuda.Enabled = false;
                this.button_loadOnnxModel.Enabled = false;
                this.button_loadLlamaModel.Enabled = false;
            }));

            if (result?.Count > 0)
            {
                MessageBox.Show($"{result.Count} Llama CUDA dependencies are installed.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                this.BeginInvoke((MethodInvoker) (() =>
                {
                    this.button_backendsLlama.Enabled = true;
                    this.button_backendsOnnx.Enabled = true;
                    this.button_loadLlamaModel.Enabled = true;
                    this.button_loadOnnxModel.Enabled = true;
                }));
            }
            else
            {
                MessageBox.Show("Failed to install Llama CUDA dependencies. Please check the logs for details.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                this.BeginInvoke((MethodInvoker) (() =>
                {
                    this.button_backendsLlama.Enabled = true;
                    this.button_backendsOnnx.Enabled = true;
                    this.button_downloadIOnnxCuda.Enabled = true;
                    this.button_downloadLlamaCuda.Enabled = true;
                    this.button_loadLlamaModel.Enabled = true;
                    this.button_loadOnnxModel.Enabled = true;
                }));
            }
        }

        
    }
}
