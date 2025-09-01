import tkinter as tk

class TranscriptionUI:
    def __init__(self, root, stop_callback, toggle_mute_callback, reset_log_callback, agent_name, my_name, default_font_size, agent_font_size):
        self.root = root
        self.stop_callback = stop_callback
        self.toggle_mute_callback = toggle_mute_callback
        self.reset_log_callback = reset_log_callback
        self.agent_name = agent_name
        self.my_name = my_name
        self.default_font_size = default_font_size
        self.agent_font_size = agent_font_size

        self.setup_ui()

    def setup_ui(self):
        self.root.title("Live Audio Chat")
        self.root.geometry("600x400")

        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.mute_button = tk.Button(button_frame, text="Mute Mic", command=self.toggle_mute_callback,
                                     bg="#ff6b6b", fg="white", font=("Arial", self.default_font_size, "bold"))
        self.mute_button.pack(side=tk.LEFT, padx=(0, 5))

        reset_button = tk.Button(button_frame, text="Reset Log", command=self.reset_log_callback,
                                 bg="#4ecdc4", fg="white", font=("Arial", self.default_font_size, "bold"))
        reset_button.pack(side=tk.LEFT, padx=(0, 5))

        self.chat_window = tk.Text(self.root, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_window.pack(expand=True, fill=tk.BOTH, padx=5, pady=(0, 5))
        self.chat_window.tag_config("microphone", foreground="blue")
        self.chat_window.tag_config("output", foreground="green")
        self.chat_window.tag_config("agent", foreground="gray", font=("Arial", self.agent_font_size, "bold"))

        self.root.protocol("WM_DELETE_WINDOW", self.stop_callback)

    def update_chat(self, transcription):
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.delete("1.0", tk.END)
        last_user = ''
        for entry in transcription:
            user, text, start_time = entry
            if user != last_user:
                if user == self.my_name:
                    self.chat_window.insert(tk.END, f"\n# {user}:\n{text}\n", "microphone")
                elif user == self.agent_name:
                    self.chat_window.insert(tk.END, f"\n# {user}:\n{text}\n", "agent")
                else:
                    self.chat_window.insert(tk.END, f"\n# {user}:\n{text}\n", "output")
                last_user = user
            else:
                if user == self.my_name:
                    self.chat_window.insert(tk.END, f"{text}\n", "microphone")
                elif user == self.agent_name:
                    self.chat_window.insert(tk.END, f"{text}\n", "agent")
                else:
                    self.chat_window.insert(tk.END, f"{text}\n", "output")

        self.chat_window.config(state=tk.DISABLED)
        self.chat_window.yview(tk.END)

    def update_mute_button(self, muted):
        if muted:
            self.root.title("Live Audio Chat - MIC MUTED")
            self.mute_button.config(text="Unmute Mic", bg="#2ecc40")
        else:
            self.root.title("Live Audio Chat")
            self.mute_button.config(text="Mute Mic", bg="#ff6b6b")
