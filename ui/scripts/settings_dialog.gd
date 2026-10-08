class_name SettingsDialog
extends Window
## Settings, same options as the Python version (minus the OpenAI assistant),
## plus the three AI roles, each with its own server, key and knowledge:
## transcription (speech to text), checks (Laya / Ollama, live mood and yes/no
## checks) and answers (a chat model that drafts answers to questions).

signal save_requested(values: Dictionary)
## `server` carries the role ("llm" = checks, "answer") and its server type, URL,
## model and key variable as currently set in the form (saved or not), as llm_* keys.
signal llm_action(action: String, server: Dictionary)
## Test the transcription server with the values in the form.
signal transcribe_check(values: Dictionary)

const KEY_ENV_HINT := "Name of an environment variable that holds the key, e.g. OPENAI_API_KEY. Leave empty for local servers."

# [tab, key, label, type, extra]
const FIELDS := [
	["General", "your_name", "Your name", "string", ""],
	["General", "languages", "Language(s)", "string", "One code or a comma list, e.g. en,lv. The first is the default."],
	["General", "auto_start", "Start transcribing when the app opens", "bool", ""],
	["General", "remote_speaker_name", "Name for unknown remote speaker", "string", "Used when Teams can't tell who talks (always on Mac)."],
	["General", "teams_window_name", "Teams window match (Windows)", "string", ""],
	["Transcription AI", "transcribe_base_url", "Server URL", "string", "Empty = OpenAI. Or any OpenAI-compatible speech-to-text server, with /v1, e.g. a local Speaches / faster-whisper server: http://127.0.0.1:8000/v1"],
	["Transcription AI", "transcribe_api_key_env", "API key environment variable", "string", KEY_ENV_HINT],
	["Transcription AI", "openai_api_key", "API key (optional, instead of the variable)", "secret", "Saved in the settings file. The variable above is used when this is empty."],
	["Transcription AI", "transcript_model", "Model", "string", "OpenAI: gpt-4o-mini-transcribe or gpt-4o-transcribe. Local servers: their model name, e.g. Systran/faster-whisper-large-v3"],
	["Transcription AI", "keywords", "Keywords (what it should know)", "multiline", "Names, product words and jargon it should expect, comma separated. They go into every request as a hint."],
	["Transcription AI", "transcribe_timeout_seconds", "Timeout (s)", "float", ""],
	["Transcription AI", "transcribe_max_retries", "Max retries", "int", ""],
	["Transcription AI", "transcribe_retry_base_seconds", "Retry base (s)", "float", ""],
	["Audio", "record_seconds", "Max chunk length (s)", "int", ""],
	["Audio", "silence_threshold", "Silence threshold", "float", "RMS level under which audio counts as silence."],
	["Audio", "silence_duration", "Silence duration (s)", "float", "How long a pause splits a chunk."],
	["Audio", "frame_duration_ms", "Frame duration (ms)", "int", ""],
	["Folders", "output_dir", "Transcriptions and profiles", "dir", ""],
	["Folders", "temp_dir", "Temporary audio", "dir", ""],
	["Filtering", "filter_min_chars", "Min chars", "int", ""],
	["Filtering", "filter_exact", "Exact matches", "string", "Comma separated lines to drop."],
	["Filtering", "filter_prefixes", "Prefix matches", "string", ""],
	["Filtering", "filter_contains", "Contains matches", "string", ""],
	["Filtering", "filter_regex", "Regex patterns", "string", ""],
	["Checks AI", "llm_enabled", "Enable live checks (mood, facts, your checks)", "bool", ""],
	["Checks AI", "llm_api", "Server type", "option", "laya,ollama,openai"],
	["Checks AI", "llm_base_url", "Server URL", "string", "Laya: http://127.0.0.1:8765. Ollama: http://127.0.0.1:11434. llama.cpp / LM Studio / cloud: their OpenAI-style URL."],
	["Checks AI", "llm_model", "Model", "string", ""],
	["Checks AI", "llm_api_key_env", "API key environment variable", "string", KEY_ENV_HINT],
	["Checks AI", "llm_context", "Extra context (what it should know)", "multiline", "Sent with every check: who is in your meetings, what the project is, words that mean something special to you."],
	["Checks AI", "llm_timeout_seconds", "Timeout (s)", "float", ""],
	["Checks AI", "llm_context_lines", "Context lines", "int", "Earlier lines sent with each check."],
	["Checks AI", "mood_enabled", "Mood of each speaker", "bool", ""],
	["Checks AI", "mood_prompt", "Mood instruction", "multiline", ""],
	["Checks AI", "fact_check_enabled", "Fact check", "bool", ""],
	["Checks AI", "fact_check_applies_to", "Fact check who", "option", "everyone,others,me"],
	["Checks AI", "fact_check_prompt", "Fact check instruction", "multiline", ""],
	["Answer AI", "answer_enabled", "Draft answers to questions people ask", "bool", ""],
	["Answer AI", "answer_api", "Server type", "option", "ollama,openai"],
	["Answer AI", "answer_base_url", "Server URL", "string", "Ollama: http://127.0.0.1:11434. LM Studio / llama.cpp: their OpenAI-style URL. OpenAI: https://api.openai.com/v1"],
	["Answer AI", "answer_model", "Model", "string", ""],
	["Answer AI", "answer_api_key_env", "API key environment variable", "string", KEY_ENV_HINT],
	["Answer AI", "answer_context", "Extra context (what it should know)", "multiline", "Facts it can answer from: your role, the project, numbers, links. The more here, the better the answers."],
	["Answer AI", "answer_prompt", "Answer instruction", "multiline", ""],
	["Answer AI", "answer_applies_to", "Answer questions from", "option", "others,everyone,me"],
	["Answer AI", "answer_timeout_seconds", "Timeout (s)", "float", ""],
	["Logging", "log_level", "Log level", "option", "DEBUG,INFO,WARNING,ERROR"],
	["Logging", "log_file_max_mb", "Log file max MB", "float", ""],
	["Logging", "log_file_backup_count", "Log backups", "int", ""],
]

## Prefix of each AI role's settings keys.
const ROLES := {"llm": "llm_", "answer": "answer_"}

var _widgets := {}
var _tabs: TabContainer
var _checks_box: VBoxContainer
## Per role ("llm", "answer"): the setup box widgets and its busy state.
var _setup := {}
var _transcribe_status: Label
const LAYA_URL := "http://127.0.0.1:8765"
const OLLAMA_URL := "http://127.0.0.1:11434"
const OLLAMA_MODELS := [
	["llama3.2:3b", "Llama 3.2 3B, about 2 GB, fast, good English"],
	["qwen2.5:3b", "Qwen 2.5 3B, about 2 GB, better with other languages"],
	["gemma3:4b", "Gemma 3 4B, about 3 GB, slower, best fact checks of the three"],
	["qwen2.5:7b", "Qwen 2.5 7B, about 5 GB, slower, better answers"],
]
const LLM_LOG_MAX_LINES := 400
var _key_set := false
var _dir_dialog: FileDialog
var _dir_target: LineEdit
var _config_path: Label


func _ready() -> void:
	title = "Settings"
	size = Vector2i(760, 680)
	min_size = Vector2i(620, 520)
	visible = false
	close_requested.connect(hide)

	var bg := PanelContainer.new()
	bg.add_theme_stylebox_override("panel", Palette.panel_style(Palette.BG, 0, 12))
	bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(bg)
	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 10)
	bg.add_child(root)

	_tabs = TabContainer.new()
	_tabs.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(_tabs)

	var tab_forms := {}
	for f in FIELDS:
		var tab: String = f[0]
		if not tab_forms.has(tab):
			tab_forms[tab] = _make_tab(tab)
		_add_field(tab_forms[tab], f)
		# The setup box sits right under server type / URL / model, which it acts on.
		if f[1] == "llm_model":
			_add_llm_controls(tab_forms[tab], "llm")
		elif f[1] == "answer_model":
			_add_llm_controls(tab_forms[tab], "answer")
		elif f[1] == "transcript_model":
			_add_transcribe_check(tab_forms[tab])
	_build_checks_tab()

	var bottom := HBoxContainer.new()
	_config_path = Label.new()
	_config_path.add_theme_color_override("font_color", Palette.TEXT_DIM)
	_config_path.add_theme_font_size_override("font_size", 11)
	_config_path.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_config_path.clip_text = true
	bottom.add_child(_config_path)
	var cancel := Button.new()
	cancel.text = "Cancel"
	cancel.pressed.connect(hide)
	bottom.add_child(cancel)
	var save := Button.new()
	save.text = "Save"
	save.custom_minimum_size = Vector2(100, 0)
	save.pressed.connect(_on_save)
	bottom.add_child(save)
	root.add_child(bottom)

	_dir_dialog = FileDialog.new()
	_dir_dialog.file_mode = FileDialog.FILE_MODE_OPEN_DIR
	_dir_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_dir_dialog.use_native_dialog = true
	_dir_dialog.dir_selected.connect(func(path): if _dir_target: _dir_target.text = path)
	add_child(_dir_dialog)


func _make_tab(tab_name: String) -> VBoxContainer:
	var scroll := ScrollContainer.new()
	scroll.name = tab_name
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	_tabs.add_child(scroll)
	var form := VBoxContainer.new()
	form.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	form.add_theme_constant_override("separation", 8)
	scroll.add_child(form)
	return form


func _add_field(form: VBoxContainer, f: Array) -> void:
	var key: String = f[1]
	var label_text: String = f[2]
	var kind: String = f[3]
	var extra: String = f[4]
	var row := VBoxContainer.new()
	row.add_theme_constant_override("separation", 2)
	if kind != "bool":
		var label := Label.new()
		label.text = label_text
		row.add_child(label)
	var widget: Control
	match kind:
		"bool":
			var cb := CheckBox.new()
			cb.text = label_text
			widget = cb
		"int", "float":
			var sb := SpinBox.new()
			sb.allow_greater = true
			sb.allow_lesser = false
			sb.max_value = 100000
			sb.step = 1.0 if kind == "int" else 0.1
			sb.custom_minimum_size = Vector2(140, 0)
			widget = sb
		"option":
			var ob := OptionButton.new()
			for item in extra.split(","):
				ob.add_item(item)
			extra = ""
			widget = ob
		"multiline":
			var te := TextEdit.new()
			te.custom_minimum_size = Vector2(0, 70)
			te.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
			widget = te
		"dir":
			var h := HBoxContainer.new()
			var le := LineEdit.new()
			le.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			h.add_child(le)
			var browse := Button.new()
			browse.text = "Browse"
			browse.pressed.connect(func():
				_dir_target = le
				_dir_dialog.current_dir = le.text
				_dir_dialog.popup_centered_ratio(0.6))
			h.add_child(browse)
			var open := Button.new()
			open.text = "Open"
			open.pressed.connect(func(): OS.shell_open(le.text))
			h.add_child(open)
			row.add_child(h)
			_widgets[key] = le
			form.add_child(row)
			return
		_:
			var le := LineEdit.new()
			le.secret = kind == "secret"
			widget = le
	row.add_child(widget)
	if extra != "":
		var hint := Label.new()
		hint.text = extra
		hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		hint.add_theme_font_size_override("font_size", 11)
		hint.add_theme_color_override("font_color", Palette.TEXT_DIM)
		row.add_child(hint)
	_widgets[key] = widget
	form.add_child(row)


func _add_transcribe_check(form: VBoxContainer) -> void:
	var row := HBoxContainer.new()
	var b := Button.new()
	b.text = "Check connection"
	b.tooltip_text = "Asks the server for its model list with the URL and key above. Costs nothing."
	b.pressed.connect(func():
		set_transcribe_status("Checking...")
		transcribe_check.emit({
			"transcribe_base_url": (_widgets["transcribe_base_url"] as LineEdit).text.strip_edges(),
			"transcribe_api_key_env": (_widgets["transcribe_api_key_env"] as LineEdit).text.strip_edges(),
			"transcript_model": (_widgets["transcript_model"] as LineEdit).text.strip_edges(),
		}))
	row.add_child(b)
	_transcribe_status = Label.new()
	_transcribe_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_transcribe_status.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_transcribe_status.add_theme_color_override("font_color", Palette.TEXT_DIM)
	row.add_child(_transcribe_status)
	form.add_child(row)


func set_transcribe_status(text: String) -> void:
	if _transcribe_status != null:
		_transcribe_status.text = text


func _add_llm_controls(form: VBoxContainer, role: String) -> void:
	var prefix: String = ROLES[role]
	var ui := {"busy": false, "busy_text": "", "busy_since": 0.0, "buttons": {}}
	_setup[role] = ui
	var box := PanelContainer.new()
	box.add_theme_stylebox_override("panel", Palette.panel_style(Palette.PANEL_LIGHT, 8, 10))
	var v := VBoxContainer.new()
	box.add_child(v)
	var info := Label.new()
	info.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	info.add_theme_color_override("font_color", Palette.TEXT_DIM)
	v.add_child(info)
	ui["info"] = info
	var status := Label.new()
	status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	v.add_child(status)
	ui["status"] = status
	var progress := ProgressBar.new()
	progress.visible = false
	progress.max_value = 1.0
	progress.step = 0.001
	v.add_child(progress)
	ui["progress"] = progress
	var buttons := HFlowContainer.new()
	for action in ["llm_install", "llm_pull", "llm_start", "llm_status", "llm_test"]:
		var b := Button.new()
		b.pressed.connect(func(): llm_action.emit(action, _server_values(role)))
		buttons.add_child(b)
		ui["buttons"][action] = b
	v.add_child(buttons)
	# Everything the installer prints (uv, pip, winget, the model download), like a small terminal.
	var log_view := TextEdit.new()
	var toggle := Button.new()
	toggle.text = "Show setup log"
	toggle.toggle_mode = true
	toggle.toggled.connect(func(on: bool):
		log_view.visible = on
		toggle.text = "Hide setup log" if on else "Show setup log")
	v.add_child(toggle)
	ui["log_toggle"] = toggle
	log_view.editable = false
	log_view.visible = false
	log_view.custom_minimum_size = Vector2(0, 220)
	log_view.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	log_view.add_theme_font_override("font", _monospace())
	log_view.add_theme_font_size_override("font_size", 12)
	log_view.add_theme_color_override("background_color", Palette.BG)
	v.add_child(log_view)
	ui["log"] = log_view
	var api: OptionButton = _widgets[prefix + "api"]
	api.item_selected.connect(func(_i):
		# Switching server type with the other one's default URL still there: use this one's default.
		var url: LineEdit = _widgets[prefix + "base_url"]
		var picked := api.get_item_text(api.selected)
		if picked == "laya" and url.text.strip_edges() in ["", OLLAMA_URL]:
			url.text = LAYA_URL
		elif picked == "ollama" and url.text.strip_edges() in ["", LAYA_URL]:
			url.text = OLLAMA_URL
		if not ui["busy"]:
			set_llm_status("Not checked yet for %s. Press \"Check if it's running\"." % picked, -2.0, false, role)
		_refresh_llm_controls(role))
	var model: LineEdit = _widgets[prefix + "model"]
	model.text_changed.connect(func(_t): _refresh_llm_controls(role))
	# Ollama model picker, fills the Model field.
	var model_row: VBoxContainer = model.get_parent()
	var presets := OptionButton.new()
	presets.add_item("Pick a suggested Ollama model...")
	for preset in OLLAMA_MODELS:
		presets.add_item("%s  -  %s" % preset)
	presets.item_selected.connect(func(i):
		if i > 0:
			model.text = OLLAMA_MODELS[i - 1][0]
			presets.select(0)
			_refresh_llm_controls(role))
	model_row.add_child(presets)
	ui["presets"] = presets
	var hint := Label.new()
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hint.add_theme_font_size_override("font_size", 11)
	hint.add_theme_color_override("font_color", Palette.TEXT_DIM)
	model_row.add_child(hint)
	ui["model_hint"] = hint
	form.add_child(box)


func _build_checks_tab() -> void:
	var form := _make_tab("Custom checks")
	var intro := Label.new()
	intro.text = "Your own live checks. Each one asks the local AI a yes/no question about every new line and pops a badge on the speaker when it hits."
	intro.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	intro.add_theme_color_override("font_color", Palette.TEXT_DIM)
	form.add_child(intro)
	_checks_box = VBoxContainer.new()
	_checks_box.add_theme_constant_override("separation", 10)
	form.add_child(_checks_box)
	var add := Button.new()
	add.text = "Add check"
	add.pressed.connect(func(): _add_check_row({"name": "", "prompt": "", "applies_to": "everyone", "enabled": true, "color": "#7c8cff"}))
	form.add_child(add)


func _add_check_row(check: Dictionary) -> void:
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", Palette.panel_style(Palette.PANEL_LIGHT, 8, 10))
	var v := VBoxContainer.new()
	panel.add_child(v)
	var top := HBoxContainer.new()
	var enabled := CheckBox.new()
	enabled.button_pressed = bool(check.get("enabled", true))
	enabled.tooltip_text = "Enabled"
	top.add_child(enabled)
	var name_edit := LineEdit.new()
	name_edit.placeholder_text = "Name, e.g. Risk mentioned"
	name_edit.text = str(check.get("name", ""))
	name_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	top.add_child(name_edit)
	var who := OptionButton.new()
	for item in ["everyone", "others", "me"]:
		who.add_item(item)
	who.select(max(0, ["everyone", "others", "me"].find(str(check.get("applies_to", "everyone")))))
	top.add_child(who)
	var color := ColorPickerButton.new()
	color.color = Color(str(check.get("color", "#7c8cff")))
	color.custom_minimum_size = Vector2(36, 0)
	top.add_child(color)
	var remove := Button.new()
	remove.text = "Delete"
	remove.pressed.connect(panel.queue_free)
	top.add_child(remove)
	v.add_child(top)
	var prompt := TextEdit.new()
	prompt.placeholder_text = "Question for the AI, e.g. Does the speaker mention a risk, blocker or delay?"
	prompt.text = str(check.get("prompt", ""))
	prompt.custom_minimum_size = Vector2(0, 60)
	prompt.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	v.add_child(prompt)
	panel.set_meta("check_id", str(check.get("id", "")))
	panel.set_meta("widgets", [enabled, name_edit, who, color, prompt])
	_checks_box.add_child(panel)


func open_with(settings: Dictionary) -> void:
	_key_set = bool(settings.get("openai_api_key_set", false))
	for key in _widgets:
		var w: Control = _widgets[key]
		var value = settings.get(key)
		if w is CheckBox:
			w.button_pressed = bool(value)
		elif w is SpinBox:
			w.value = float(value if value != null else 0)
		elif w is OptionButton:
			for i in w.item_count:
				if w.get_item_text(i) == str(value):
					w.select(i)
		elif w is TextEdit:
			w.text = str(value if value != null else "")
		elif w is LineEdit:
			if key == "openai_api_key":
				w.text = ""
				w.placeholder_text = "saved (type to replace)" if _key_set else "sk-..."
			else:
				w.text = str(value if value != null else "")
	for child in _checks_box.get_children():
		child.queue_free()
	for check in settings.get("custom_checks", []):
		_add_check_row(check)
	_config_path.text = "Settings file: " + str(settings.get("config_path", ""))
	for role in _setup:
		_refresh_llm_controls(role)
	set_transcribe_status("")
	popup_centered()


func _on_save() -> void:
	var values := {}
	for key in _widgets:
		var w: Control = _widgets[key]
		if w is CheckBox:
			values[key] = w.button_pressed
		elif w is SpinBox:
			values[key] = w.value
		elif w is OptionButton:
			values[key] = w.get_item_text(w.selected)
		elif w is TextEdit:
			values[key] = w.text.strip_edges()
		elif w is LineEdit:
			values[key] = w.text.strip_edges()
	if values.get("openai_api_key", "") == "":
		values.erase("openai_api_key")
	var checks: Array = []
	for panel in _checks_box.get_children():
		if panel.is_queued_for_deletion() or not panel.has_meta("widgets"):
			continue
		var ws: Array = panel.get_meta("widgets")
		var name_text: String = ws[1].text.strip_edges()
		var prompt_text: String = ws[4].text.strip_edges()
		if name_text == "" or prompt_text == "":
			continue
		checks.append({
			"id": panel.get_meta("check_id"),
			"enabled": ws[0].button_pressed,
			"name": name_text,
			"applies_to": ws[2].get_item_text(ws[2].selected),
			"color": "#" + ws[3].color.to_html(false),
			"prompt": prompt_text,
		})
	values["custom_checks"] = checks
	save_requested.emit(values)
	hide()


func _server_values(role: String) -> Dictionary:
	var prefix: String = ROLES[role]
	var api: OptionButton = _widgets[prefix + "api"]
	return {
		"role": role,
		"llm_api": api.get_item_text(api.selected),
		"llm_base_url": (_widgets[prefix + "base_url"] as LineEdit).text.strip_edges(),
		"llm_model": (_widgets[prefix + "model"] as LineEdit).text.strip_edges(),
		"llm_api_key_env": (_widgets[prefix + "api_key_env"] as LineEdit).text.strip_edges(),
	}


## Say exactly what each button does for the server type picked right now.
func _refresh_llm_controls(role: String) -> void:
	if not _setup.has(role):
		return
	var ui: Dictionary = _setup[role]
	var server := _server_values(role)
	var model: String = server["llm_model"]
	var model_edit: LineEdit = _widgets[ROLES[role] + "model"]
	var info: Label = ui["info"]
	var hint: Label = ui["model_hint"]
	var presets: OptionButton = ui["presets"]
	var b: Dictionary = ui["buttons"]
	match server["llm_api"]:
		"laya":
			info.text = ("Laya is a small local decision model (the pip package \"laya\"). It answers the mood and yes/no " +
				"checks for each line in one fast pass, in 100+ languages, but for fact checks it can only flag a line " +
				"as probably wrong. Nothing leaves your computer.")
			b["llm_install"].text = "Install Laya (about 1 GB)"
			b["llm_install"].tooltip_text = ("Downloads uv, which brings its own Python, then installs Laya and PyTorch " +
				"into the app's settings folder (laya/venv), starts the Laya server and downloads the model. " +
				"Nothing else on your computer is changed.")
			b["llm_pull"].text = "Download Laya model"
			b["llm_pull"].tooltip_text = ("Downloads Laya's English and multilingual checkpoints from Hugging Face " +
				"(convaiinnovations/laya). Laya picks one per line by its language. Install already does this.")
			b["llm_start"].text = "Start Laya server"
			b["llm_start"].tooltip_text = "Starts the Laya server in the background (the app also starts it by itself when it opens)."
			model_edit.visible = false
			hint.text = "Laya has no model to pick: it chooses its English or multilingual checkpoint per line."
			presets.visible = false
		"ollama":
			if role == "answer":
				info.text = ("Ollama runs a chat model on your computer that writes the answers. A bigger model answers " +
					"better but slower. Nothing leaves your computer.")
			else:
				info.text = ("Ollama runs a full chat model on your computer. Slower than Laya, but fact checks come with " +
					"a short explanation. Nothing leaves your computer.")
			b["llm_install"].text = "Install Ollama"
			b["llm_install"].tooltip_text = ("Installs the Ollama app (winget on Windows, Homebrew or the app download " +
				"on macOS), starts it and downloads the model below.")
			b["llm_pull"].text = "Download %s" % (model if model != "" else "model")
			b["llm_pull"].tooltip_text = "Downloads the model named in Model from ollama.com into Ollama."
			b["llm_start"].text = "Start Ollama server"
			b["llm_start"].tooltip_text = "Starts Ollama in the background if it isn't running."
			model_edit.visible = true
			hint.text = "The Ollama model to use and download. Pick one of the suggestions or type any name from ollama.com/library."
			presets.visible = true
		_:
			info.text = ("Any OpenAI-compatible server: local (LM Studio, llama.cpp server) or in the cloud (OpenAI, Groq...). " +
				"You run or sign up for it yourself; the app connects to the URL with the model below, and the key " +
				"from the environment variable if one is set. Cloud servers see the transcript lines.")
			model_edit.visible = true
			hint.text = "The model name your server expects."
			presets.visible = false
	var managed: bool = server["llm_api"] != "openai"
	b["llm_install"].visible = managed
	b["llm_pull"].visible = managed
	b["llm_start"].visible = managed
	b["llm_status"].text = "Check if it's running"
	b["llm_status"].tooltip_text = "Asks the server at the URL above whether it's up and the model is there."
	b["llm_test"].text = "Test with a sample question" if role == "answer" else "Test with a sample line"
	b["llm_test"].tooltip_text = "Sends one sample and shows the answer and how long it took."


func _monospace() -> SystemFont:
	var font := SystemFont.new()
	font.font_names = PackedStringArray(["Consolas", "Menlo", "DejaVu Sans Mono", "monospace"])
	return font


func _ui(role: String) -> Dictionary:
	return _setup.get(role, _setup.get("llm", {}))


func show_llm_log(role: String = "llm") -> void:
	var ui := _ui(role)
	if not ui.is_empty() and not ui["log_toggle"].button_pressed:
		ui["log_toggle"].button_pressed = true


func append_llm_log(line: String, role: String = "llm") -> void:
	var ui := _ui(role)
	if ui.is_empty():
		return
	var log_view: TextEdit = ui["log"]
	if log_view.text != "":
		log_view.text += "\n"
	log_view.text += line
	if log_view.get_line_count() > LLM_LOG_MAX_LINES:
		var lines := log_view.text.split("\n")
		log_view.text = "\n".join(lines.slice(lines.size() - LLM_LOG_MAX_LINES))
	_scroll_log_to_end.call_deferred(log_view)


func _scroll_log_to_end(log_view: TextEdit) -> void:
	log_view.set_caret_line(log_view.get_line_count() - 1)
	log_view.adjust_viewport_to_caret()


func set_llm_log(lines: Array, role: String = "llm") -> void:
	var ui := _ui(role)
	if ui.is_empty():
		return
	ui["log"].text = ""
	for line in lines:
		append_llm_log(str(line), role)


## While a setup task runs, keep the elapsed time ticking even when the installer is quiet.
func set_llm_busy(text: String, progress: float, elapsed: float, role: String = "llm") -> void:
	var ui := _ui(role)
	if ui.is_empty():
		return
	if not ui["busy"]:
		show_llm_log(role)
	ui["busy"] = true
	ui["busy_text"] = text
	ui["busy_since"] = Time.get_ticks_msec() / 1000.0 - elapsed
	set_llm_status(_busy_label(ui), progress, true, role)


func _busy_label(ui: Dictionary) -> String:
	var seconds := maxi(0, int(Time.get_ticks_msec() / 1000.0 - float(ui["busy_since"])))
	return "%s  (%d:%02d)" % [ui["busy_text"], seconds / 60, seconds % 60]


var _llm_tick := 0.0


func _process(delta: float) -> void:
	_llm_tick += delta
	if _llm_tick < 1.0:
		return
	_llm_tick = 0.0
	for role in _setup:
		var ui: Dictionary = _setup[role]
		if ui["busy"]:
			ui["status"].text = _busy_label(ui)


## Server status from a Check; while a setup task runs its progress line wins.
func set_llm_info(text: String, role: String = "llm") -> void:
	var ui := _ui(role)
	if not ui.is_empty() and not ui["busy"]:
		set_llm_status(text, -2.0, false, role)


func set_llm_status(text: String, progress: float = -2.0, busy: bool = false, role: String = "llm") -> void:
	var ui := _ui(role)
	if ui.is_empty():
		return
	if not busy:
		ui["busy"] = false
	ui["status"].text = text
	var bar: ProgressBar = ui["progress"]
	bar.visible = progress > -2.0
	if progress >= 0.0:
		bar.value = progress
		bar.indeterminate = false
	elif progress > -2.0:
		bar.indeterminate = true
