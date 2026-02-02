namespace OnnxPeriment.Forms
{
    partial class WindowMain
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.listBox_log = new ListBox();
            this.button_loadOnnxModel = new Button();
            this.checkBox_enableCuda = new CheckBox();
            this.pictureBox_view = new PictureBox();
            this.numericUpDown_images = new NumericUpDown();
            this.button_importImage = new Button();
            this.button_deleteImage = new Button();
            this.label_imageInfo = new Label();
            this.checkBox_zoomImage = new CheckBox();
            this.textBox_prompt = new TextBox();
            this.button_send = new Button();
            this.textBox_response = new TextBox();
            this.checkBox_includeImage = new CheckBox();
            this.label_timer = new Label();
            this.button_loadLlamaModel = new Button();
            this.label_status = new Label();
            this.button_backendsOnnx = new Button();
            this.button_backendsLlama = new Button();
            this.button_downloadIOnnxCuda = new Button();
            this.button_downloadLlamaCuda = new Button();
            this.label_stats = new Label();
            this.label_context = new Label();
            this.button_loadContext = new Button();
            this.numericUpDown_messages = new NumericUpDown();
            ((System.ComponentModel.ISupportInitialize) this.pictureBox_view).BeginInit();
            ((System.ComponentModel.ISupportInitialize) this.numericUpDown_images).BeginInit();
            ((System.ComponentModel.ISupportInitialize) this.numericUpDown_messages).BeginInit();
            this.SuspendLayout();
            // 
            // listBox_log
            // 
            this.listBox_log.FormattingEnabled = true;
            this.listBox_log.Location = new Point(12, 465);
            this.listBox_log.Name = "listBox_log";
            this.listBox_log.Size = new Size(520, 244);
            this.listBox_log.TabIndex = 0;
            this.listBox_log.DoubleClick += this.listBox_log_DoubleClick;
            // 
            // button_loadOnnxModel
            // 
            this.button_loadOnnxModel.Location = new Point(538, 628);
            this.button_loadOnnxModel.Name = "button_loadOnnxModel";
            this.button_loadOnnxModel.Size = new Size(75, 23);
            this.button_loadOnnxModel.TabIndex = 1;
            this.button_loadOnnxModel.Text = "ONNX";
            this.button_loadOnnxModel.UseVisualStyleBackColor = true;
            this.button_loadOnnxModel.Click += this.button_loadOnnxModel_Click;
            // 
            // checkBox_enableCuda
            // 
            this.checkBox_enableCuda.AutoSize = true;
            this.checkBox_enableCuda.Checked = true;
            this.checkBox_enableCuda.CheckState = CheckState.Checked;
            this.checkBox_enableCuda.Location = new Point(538, 690);
            this.checkBox_enableCuda.Name = "checkBox_enableCuda";
            this.checkBox_enableCuda.Size = new Size(96, 19);
            this.checkBox_enableCuda.TabIndex = 2;
            this.checkBox_enableCuda.Text = "enable CUDA";
            this.checkBox_enableCuda.UseVisualStyleBackColor = true;
            // 
            // pictureBox_view
            // 
            this.pictureBox_view.BackColor = SystemColors.ActiveBorder;
            this.pictureBox_view.Location = new Point(732, 12);
            this.pictureBox_view.Name = "pictureBox_view";
            this.pictureBox_view.Size = new Size(480, 360);
            this.pictureBox_view.TabIndex = 3;
            this.pictureBox_view.TabStop = false;
            // 
            // numericUpDown_images
            // 
            this.numericUpDown_images.Location = new Point(1152, 378);
            this.numericUpDown_images.Maximum = new decimal(new int[] { 0, 0, 0, 0 });
            this.numericUpDown_images.Name = "numericUpDown_images";
            this.numericUpDown_images.Size = new Size(60, 23);
            this.numericUpDown_images.TabIndex = 4;
            this.numericUpDown_images.ValueChanged += this.numericUpDown_images_ValueChanged;
            // 
            // button_importImage
            // 
            this.button_importImage.Location = new Point(651, 260);
            this.button_importImage.Name = "button_importImage";
            this.button_importImage.Size = new Size(75, 23);
            this.button_importImage.TabIndex = 5;
            this.button_importImage.Text = "Import";
            this.button_importImage.UseVisualStyleBackColor = true;
            this.button_importImage.Click += this.button_importImage_Click;
            // 
            // button_deleteImage
            // 
            this.button_deleteImage.Location = new Point(651, 349);
            this.button_deleteImage.Name = "button_deleteImage";
            this.button_deleteImage.Size = new Size(75, 23);
            this.button_deleteImage.TabIndex = 6;
            this.button_deleteImage.Text = "Delete";
            this.button_deleteImage.UseVisualStyleBackColor = true;
            this.button_deleteImage.Click += this.button_deleteImage_Click;
            // 
            // label_imageInfo
            // 
            this.label_imageInfo.AutoSize = true;
            this.label_imageInfo.Location = new Point(732, 378);
            this.label_imageInfo.Name = "label_imageInfo";
            this.label_imageInfo.Size = new Size(137, 15);
            this.label_imageInfo.TabIndex = 7;
            this.label_imageInfo.Text = "No image data available.";
            // 
            // checkBox_zoomImage
            // 
            this.checkBox_zoomImage.AutoSize = true;
            this.checkBox_zoomImage.Checked = true;
            this.checkBox_zoomImage.CheckState = CheckState.Checked;
            this.checkBox_zoomImage.Location = new Point(1152, 407);
            this.checkBox_zoomImage.Name = "checkBox_zoomImage";
            this.checkBox_zoomImage.Size = new Size(58, 19);
            this.checkBox_zoomImage.TabIndex = 8;
            this.checkBox_zoomImage.Text = "Zoom";
            this.checkBox_zoomImage.UseVisualStyleBackColor = true;
            this.checkBox_zoomImage.CheckedChanged += this.checkBox_centerImage_CheckedChanged;
            // 
            // textBox_prompt
            // 
            this.textBox_prompt.Location = new Point(732, 594);
            this.textBox_prompt.MaxLength = 65536;
            this.textBox_prompt.Multiline = true;
            this.textBox_prompt.Name = "textBox_prompt";
            this.textBox_prompt.PlaceholderText = "Enter prompt here ...";
            this.textBox_prompt.Size = new Size(480, 86);
            this.textBox_prompt.TabIndex = 9;
            this.textBox_prompt.KeyDown += this.textBox_prompt_KeyDown;
            // 
            // button_send
            // 
            this.button_send.Location = new Point(1167, 686);
            this.button_send.Name = "button_send";
            this.button_send.Size = new Size(45, 23);
            this.button_send.TabIndex = 10;
            this.button_send.Text = "Send";
            this.button_send.UseVisualStyleBackColor = true;
            this.button_send.Click += this.button_send_Click;
            // 
            // textBox_response
            // 
            this.textBox_response.Location = new Point(732, 432);
            this.textBox_response.MaxLength = 99999999;
            this.textBox_response.Multiline = true;
            this.textBox_response.Name = "textBox_response";
            this.textBox_response.PlaceholderText = "Model response will be put here ...";
            this.textBox_response.ReadOnly = true;
            this.textBox_response.ScrollBars = ScrollBars.Vertical;
            this.textBox_response.Size = new Size(480, 133);
            this.textBox_response.TabIndex = 11;
            // 
            // checkBox_includeImage
            // 
            this.checkBox_includeImage.AutoSize = true;
            this.checkBox_includeImage.Location = new Point(1077, 689);
            this.checkBox_includeImage.Name = "checkBox_includeImage";
            this.checkBox_includeImage.Size = new Size(84, 19);
            this.checkBox_includeImage.TabIndex = 12;
            this.checkBox_includeImage.Text = "Add Image";
            this.checkBox_includeImage.UseVisualStyleBackColor = true;
            // 
            // label_timer
            // 
            this.label_timer.AutoSize = true;
            this.label_timer.Location = new Point(732, 576);
            this.label_timer.Name = "label_timer";
            this.label_timer.Size = new Size(25, 15);
            this.label_timer.TabIndex = 13;
            this.label_timer.Text = "-:--";
            // 
            // button_loadLlamaModel
            // 
            this.button_loadLlamaModel.Location = new Point(538, 657);
            this.button_loadLlamaModel.Name = "button_loadLlamaModel";
            this.button_loadLlamaModel.Size = new Size(75, 23);
            this.button_loadLlamaModel.TabIndex = 14;
            this.button_loadLlamaModel.Text = "Llama";
            this.button_loadLlamaModel.UseVisualStyleBackColor = true;
            this.button_loadLlamaModel.Click += this.button_loadLlamaModel_Click;
            // 
            // label_status
            // 
            this.label_status.AutoSize = true;
            this.label_status.Location = new Point(732, 683);
            this.label_status.Name = "label_status";
            this.label_status.Size = new Size(102, 15);
            this.label_status.TabIndex = 15;
            this.label_status.Text = "No model loaded.";
            // 
            // button_backendsOnnx
            // 
            this.button_backendsOnnx.Location = new Point(619, 628);
            this.button_backendsOnnx.Name = "button_backendsOnnx";
            this.button_backendsOnnx.Size = new Size(65, 23);
            this.button_backendsOnnx.TabIndex = 16;
            this.button_backendsOnnx.Text = "Backends";
            this.button_backendsOnnx.UseVisualStyleBackColor = true;
            this.button_backendsOnnx.Click += this.button_backendsOnnx_Click;
            // 
            // button_backendsLlama
            // 
            this.button_backendsLlama.Location = new Point(619, 657);
            this.button_backendsLlama.Name = "button_backendsLlama";
            this.button_backendsLlama.Size = new Size(65, 23);
            this.button_backendsLlama.TabIndex = 17;
            this.button_backendsLlama.Text = "Backends";
            this.button_backendsLlama.UseVisualStyleBackColor = true;
            this.button_backendsLlama.Click += this.button_backendsLlama_Click;
            // 
            // button_downloadIOnnxCuda
            // 
            this.button_downloadIOnnxCuda.Location = new Point(690, 628);
            this.button_downloadIOnnxCuda.Name = "button_downloadIOnnxCuda";
            this.button_downloadIOnnxCuda.Size = new Size(36, 23);
            this.button_downloadIOnnxCuda.TabIndex = 18;
            this.button_downloadIOnnxCuda.Text = "DL";
            this.button_downloadIOnnxCuda.UseVisualStyleBackColor = true;
            this.button_downloadIOnnxCuda.Click += this.button_downloadIOnnxCuda_Click;
            // 
            // button_downloadLlamaCuda
            // 
            this.button_downloadLlamaCuda.Location = new Point(690, 657);
            this.button_downloadLlamaCuda.Name = "button_downloadLlamaCuda";
            this.button_downloadLlamaCuda.Size = new Size(36, 23);
            this.button_downloadLlamaCuda.TabIndex = 19;
            this.button_downloadLlamaCuda.Text = "DL";
            this.button_downloadLlamaCuda.UseVisualStyleBackColor = true;
            this.button_downloadLlamaCuda.Click += this.button_downloadLlamaCuda_Click;
            // 
            // label_stats
            // 
            this.label_stats.AutoSize = true;
            this.label_stats.Location = new Point(796, 568);
            this.label_stats.Name = "label_stats";
            this.label_stats.Size = new Size(102, 15);
            this.label_stats.TabIndex = 20;
            this.label_stats.Text = "No stats available.";
            // 
            // label_context
            // 
            this.label_context.AutoSize = true;
            this.label_context.Location = new Point(732, 414);
            this.label_context.Name = "label_context";
            this.label_context.Size = new Size(107, 15);
            this.label_context.TabIndex = 21;
            this.label_context.Text = "No context loaded.";
            // 
            // button_loadContext
            // 
            this.button_loadContext.Location = new Point(651, 410);
            this.button_loadContext.Name = "button_loadContext";
            this.button_loadContext.Size = new Size(75, 23);
            this.button_loadContext.TabIndex = 22;
            this.button_loadContext.Text = "Load JSON";
            this.button_loadContext.UseVisualStyleBackColor = true;
            this.button_loadContext.Click += this.button_loadContext_Click;
            // 
            // numericUpDown_messages
            // 
            this.numericUpDown_messages.Location = new Point(651, 439);
            this.numericUpDown_messages.Maximum = new decimal(new int[] { 0, 0, 0, 0 });
            this.numericUpDown_messages.Name = "numericUpDown_messages";
            this.numericUpDown_messages.Size = new Size(75, 23);
            this.numericUpDown_messages.TabIndex = 23;
            this.numericUpDown_messages.ValueChanged += this.numericUpDown_messages_ValueChanged;
            this.numericUpDown_messages.Click += this.numericUpDown_messages_Click;
            // 
            // WindowMain
            // 
            this.AutoScaleDimensions = new SizeF(7F, 15F);
            this.AutoScaleMode = AutoScaleMode.Font;
            this.ClientSize = new Size(1224, 721);
            this.Controls.Add(this.numericUpDown_messages);
            this.Controls.Add(this.button_loadContext);
            this.Controls.Add(this.label_context);
            this.Controls.Add(this.label_stats);
            this.Controls.Add(this.button_downloadLlamaCuda);
            this.Controls.Add(this.button_downloadIOnnxCuda);
            this.Controls.Add(this.button_backendsLlama);
            this.Controls.Add(this.button_backendsOnnx);
            this.Controls.Add(this.label_status);
            this.Controls.Add(this.button_loadLlamaModel);
            this.Controls.Add(this.label_timer);
            this.Controls.Add(this.checkBox_includeImage);
            this.Controls.Add(this.textBox_response);
            this.Controls.Add(this.button_send);
            this.Controls.Add(this.textBox_prompt);
            this.Controls.Add(this.checkBox_zoomImage);
            this.Controls.Add(this.label_imageInfo);
            this.Controls.Add(this.button_deleteImage);
            this.Controls.Add(this.button_importImage);
            this.Controls.Add(this.numericUpDown_images);
            this.Controls.Add(this.pictureBox_view);
            this.Controls.Add(this.checkBox_enableCuda);
            this.Controls.Add(this.button_loadOnnxModel);
            this.Controls.Add(this.listBox_log);
            this.Name = "WindowMain";
            this.Text = "Onnx Experiment (Forms UI)";
            ((System.ComponentModel.ISupportInitialize) this.pictureBox_view).EndInit();
            ((System.ComponentModel.ISupportInitialize) this.numericUpDown_images).EndInit();
            ((System.ComponentModel.ISupportInitialize) this.numericUpDown_messages).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();
        }

        #endregion

        private ListBox listBox_log;
        private Button button_loadOnnxModel;
        private CheckBox checkBox_enableCuda;
        private PictureBox pictureBox_view;
        private NumericUpDown numericUpDown_images;
        private Button button_importImage;
        private Button button_deleteImage;
        private Label label_imageInfo;
        private CheckBox checkBox_zoomImage;
        private TextBox textBox_prompt;
        private Button button_send;
        private TextBox textBox_response;
        private CheckBox checkBox_includeImage;
        private Label label_timer;
        private Button button_loadLlamaModel;
        private Label label_status;
        private Button button_backendsOnnx;
        private Button button_backendsLlama;
        private Button button_downloadIOnnxCuda;
        private Button button_downloadLlamaCuda;
        private Label label_stats;
        private Label label_context;
        private Button button_loadContext;
        private NumericUpDown numericUpDown_messages;
    }
}
