class_name SettingsDialog
extends Window
## Settings, same options as the Python version (minus the OpenAI assistant),
## plus Local AI (Ollama) and user-defined live checks.

signal save_requested(values: Dictionary)
## `server` carries the server type, URL and model as currently set in the form (saved or not).
signal llm_action(action: String, server: Dictionary)

# [tab, key, label, type, extra]
const FIELDS := [
	["General", "your_name", "Your name", "string", ""],
	["General", "languages", "Language(s)", "string", "One code or a comma list, e.g. en,lv. The first is the default."],
	["General", "keywords", "Keywords", "string", "Hard words for the transcriber, comma separated."],
	["General", "auto_start", "Start transcribing when the app opens", "bool", ""],
	["General", "remote_speaker_name", "Name for unknown remote speaker", "string", "Used when Teams can't tell who talks (always on Mac)."],
	["General", "teams_window_name", "Teams window match (Windows)", "string", ""],
	["Transcription", "openai_api_key", "OpenAI API key", "secret", ""],
	["Transcription", "transcript_model", "Transcription model", "string", "gpt-4o-mini-transcribe or gpt-4o-transcribe"],
	["Transcription", "transcribe_timeout_seconds", "Timeout (s)", "float", ""],
	["Transcription", "transcribe_max_retries", "Max retries", "int", ""],
	["Transcription", "transcribe_retry_base_seconds", "Retry base (s)", "float", ""],
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
	["Local AI", "llm_enabled", "Enable live checks on local AI", "bool", ""],
	["Local AI", "llm_api", "Server type", "option", "laya,ollama,openai"],
	["Local AI", "llm_base_url", "Server URL", "string", "Laya: http://127.0.0.1:8765. Ollama: http://127.0.0.1:11434. llama.cpp / LM Studio: their OpenAI-style URL."],
	["Local AI", "llm_model", "Model", "string", ""],
	["Local AI", "llm_timeout_seconds", "Timeout (s)", "float", ""],
	["Local AI", "llm_context_lines", "Context lines", "int", "Earlier lines sent with each check."],
	["Local AI", "mood_enabled", "Mood of each speaker", "bool", ""],
	["Local AI", "mood_prompt", "Mood instruction", "multiline", ""],
	["Local AI", "fact_check_enabled", "Fact check", "bool", ""],
	["Local AI", "fact_check_applies_to", "Fact check who", "option", "everyone,others,me"],
	["Local AI", "fact_check_prompt", "Fact check instruction", "multiline", ""],
	["Logging", "log_level", "Log level", "option", "DEBUG,INFO,WARNING,ERROR"],
	["Logging", "log_file_max_mb", "Log file max MB", "float", ""],
	["Logging", "log_file_backup_count", "Log backups", "int", ""],
]

var _widgets := {}
var _tabs: TabContainer
var _checks_box: VBoxContainer
var _llm_status: Label
var _llm_progress: ProgressBar
var _llm_log: TextEdit
var _llm_info: Label
var _llm_buttons := {}
var _model_hint: Label
var _model_presets: OptionButton
const LAYA_URL := "http://127.0.0.1:8765"
const OLLAMA_URL := "http://127.0.0.1:11434"
const OLLAMA_MODELS := [
	["llama3.2:3b", "Llama 3.2 3B, about 2 GB, fast, good English"],
	["qwen2.5:3b", "Qwen 2.5 3B, about 2 GB, better with other languages"],
	["gemma3:4b", "Gemma 3 4B, about 3 GB, slower, best fact checks of the three"],
]
var _llm_log_toggle: Button
var _llm_busy_text := ""
var _llm_busy := false
var _llm_busy_since := 0.0
var _llm_tick := 0.0
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
			_add_llm_controls(tab_forms[tab])
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


func _add_llm_controls(form: VBoxContainer) -> void:
	var box := PanelContainer.new()
	box.add_theme_stylebox_override("panel", Palette.panel_style(Palette.PANEL_LIGHT, 8, 10))
	var v := VBoxContainer.new()
	box.add_child(v)
	_llm_info = Label.new()
	_llm_info.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_llm_info.add_theme_color_override("font_color", Palette.TEXT_DIM)
	v.add_child(_llm_info)
	_llm_status = Label.new()
	_llm_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	v.add_child(_llm_status)
	_llm_progress = ProgressBar.new()
	_llm_progress.visible = false
	_llm_progress.max_value = 1.0
	_llm_progress.step = 0.001
	v.add_child(_llm_progress)
	var buttons := HFlowContainer.new()
	for action in ["llm_install", "llm_pull", "llm_start", "llm_status", "llm_test"]:
		var b := Button.new()
		b.pressed.connect(func(): llm_action.emit(action, _server_values()))
		buttons.add_child(b)
		_llm_buttons[action] = b
	v.add_child(buttons)
	# Everything the installer prints (uv, pip, winget, the model download), like a small terminal.
	_llm_log_toggle = Button.new()
	_llm_log_toggle.text = "Show setup log"
	_llm_log_toggle.toggle_mode = true
	_llm_log_toggle.toggled.connect(func(on: bool):
		_llm_log.visible = on
		_llm_log_toggle.text = "Hide setup log" if on else "Show setup log")
	v.add_child(_llm_log_toggle)
	_llm_log = TextEdit.new()
	_llm_log.editable = false
	_llm_log.visible = false
	_llm_log.custom_minimum_size = Vector2(0, 220)
	_llm_log.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	_llm_log.add_theme_font_override("font", _monospace())
	_llm_log.add_theme_font_size_override("font_size", 12)
	_llm_log.add_theme_color_override("background_color", Palette.BG)
	v.add_child(_llm_log)
	var api: OptionButton = _widgets["llm_api"]
	api.item_selected.connect(func(_i):
		# Switching server type with the other one's default URL still there: use this one's default.
		var url: LineEdit = _widgets["llm_base_url"]
		var picked := api.get_item_text(api.selected)
		if picked == "laya" and url.text.strip_edges() in ["", OLLAMA_URL]:
			url.text = LAYA_URL
		elif picked == "ollama" and url.text.strip_edges() in ["", LAYA_URL]:
			url.text = OLLAMA_URL
		if not _llm_busy:
			set_llm_status("Not checked yet for %s. Press \"Check if it's running\"." % picked)
		_refresh_llm_controls())
	var model: LineEdit = _widgets["llm_model"]
	model.text_changed.connect(func(_t): _refresh_llm_controls())
	# Ollama model picker, fills the Model field.
	var model_row: VBoxContainer = model.get_parent()
	_model_presets = OptionButton.new()
	_model_presets.add_item("Pick a suggested Ollama model...")
	for preset in OLLAMA_MODELS:
		_model_presets.add_item("%s  -  %s" % preset)
	_model_presets.item_selected.connect(func(i):
		if i > 0:
			model.text = OLLAMA_MODELS[i - 1][0]
			_model_presets.select(0)
			_refresh_llm_controls())
	model_row.add_child(_model_presets)
	_model_hint = Label.new()
	_model_hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_model_hint.add_theme_font_size_override("font_size", 11)
	_model_hint.add_theme_color_override("font_color", Palette.TEXT_DIM)
	model_row.add_child(_model_hint)
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
	_refresh_llm_controls()
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


func _server_values() -> Dictionary:
	var api: OptionButton = _widgets["llm_api"]
	return {
		"llm_api": api.get_item_text(api.selected),
		"llm_base_url": (_widgets["llm_base_url"] as LineEdit).text.strip_edges(),
		"llm_model": (_widgets["llm_model"] as LineEdit).text.strip_edges(),
	}


## Say exactly what each button does for the server type picked right now.
func _refresh_llm_controls() -> void:
	if _llm_info == null:
		return
	var server := _server_values()
	var model: String = server["llm_model"]
	var model_edit: LineEdit = _widgets["llm_model"]
	var b: Dictionary = _llm_buttons
	match server["llm_api"]:
		"laya":
			_llm_info.text = ("Laya is a small local decision model (the pip package \"laya\"). It answers the mood and yes/no " +
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
			_model_hint.text = "Laya has no model to pick: it chooses its English or multilingual checkpoint per line."
			_model_presets.visible = false
		"ollama":
			_llm_info.text = ("Ollama runs a full chat model on your computer. Slower than Laya, but fact checks come with " +
				"a short explanation. Nothing leaves your computer.")
			b["llm_install"].text = "Install Ollama"
			b["llm_install"].tooltip_text = ("Installs the Ollama app (winget on Windows, Homebrew or the app download " +
				"on macOS), starts it and downloads the model below.")
			b["llm_pull"].text = "Download %s" % (model if model != "" else "model")
			b["llm_pull"].tooltip_text = "Downloads the model named in Model from ollama.com into Ollama."
			b["llm_start"].text = "Start Ollama server"
			b["llm_start"].tooltip_text = "Starts Ollama in the background if it isn't running."
			model_edit.visible = true
			_model_hint.text = "The Ollama model to use and download. Pick one of the suggestions or type any name from ollama.com/library."
			_model_presets.visible = true
		_:
			_llm_info.text = ("Your own OpenAI-compatible server (LM Studio, llama.cpp server...). You install and start it " +
				"yourself; the app only connects to the URL and uses the model name below.")
			model_edit.visible = true
			_model_hint.text = "The model name your server expects."
			_model_presets.visible = false
	var managed: bool = server["llm_api"] != "openai"
	b["llm_install"].visible = managed
	b["llm_pull"].visible = managed
	b["llm_start"].visible = managed
	b["llm_status"].text = "Check if it's running"
	b["llm_status"].tooltip_text = "Asks the server at the URL above whether it's up and the model is there."
	b["llm_test"].text = "Test with a sample line"
	b["llm_test"].tooltip_text = "Sends one sample sentence and shows the answer and how long it took."


func _monospace() -> SystemFont:
	var font := SystemFont.new()
	font.font_names = PackedStringArray(["Consolas", "Menlo", "DejaVu Sans Mono", "monospace"])
	return font


func show_llm_log() -> void:
	if _llm_log_toggle != null and not _llm_log_toggle.button_pressed:
		_llm_log_toggle.button_pressed = true


func append_llm_log(line: String) -> void:
	if _llm_log == null:
		return
	if _llm_log.text != "":
		_llm_log.text += "\n"
	_llm_log.text += line
	if _llm_log.get_line_count() > LLM_LOG_MAX_LINES:
		var lines := _llm_log.text.split("\n")
		_llm_log.text = "\n".join(lines.slice(lines.size() - LLM_LOG_MAX_LINES))
	_scroll_log_to_end.call_deferred()


func _scroll_log_to_end() -> void:
	_llm_log.set_caret_line(_llm_log.get_line_count() - 1)
	_llm_log.adjust_viewport_to_caret()


func set_llm_log(lines: Array) -> void:
	if _llm_log == null:
		return
	_llm_log.text = ""
	for line in lines:
		append_llm_log(str(line))


## While a setup task runs, keep the elapsed time ticking even when the installer is quiet.
func set_llm_busy(text: String, progress: float, elapsed: float) -> void:
	if not _llm_busy:
		show_llm_log()
	_llm_busy = true
	_llm_busy_text = text
	_llm_busy_since = Time.get_ticks_msec() / 1000.0 - elapsed
	set_llm_status(_busy_label(), progress, true)


func _busy_label() -> String:
	var seconds := maxi(0, int(Time.get_ticks_msec() / 1000.0 - _llm_busy_since))
	return "%s  (%d:%02d)" % [_llm_busy_text, seconds / 60, seconds % 60]


func _process(delta: float) -> void:
	if not _llm_busy or _llm_status == null:
		return
	_llm_tick += delta
	if _llm_tick >= 1.0:
		_llm_tick = 0.0
		_llm_status.text = _busy_label()


## Server status from a Check; while a setup task runs its progress line wins.
func set_llm_info(text: String) -> void:
	if not _llm_busy:
		set_llm_status(text)


func set_llm_status(text: String, progress: float = -2.0, busy: bool = false) -> void:
	if _llm_status == null:
		return
	if not busy:
		_llm_busy = false
	_llm_status.text = text
	_llm_progress.visible = progress > -2.0
	if progress >= 0.0:
		_llm_progress.value = progress
		_llm_progress.indeterminate = false
	elif progress > -2.0:
		_llm_progress.indeterminate = true
