class_name Palette
extends RefCounted
## Shared colors and small formatting helpers.

const BG := Color("11131a")
const PANEL := Color("181b25")
const PANEL_LIGHT := Color("222634")
const BORDER := Color("2c3144")
const TEXT := Color("e6e8ef")
const TEXT_DIM := Color("8b91a7")
const ACCENT := Color("7c8cff")
const RECORDING := Color("ef476f")
const GOOD := Color("06d6a0")
const WARN := Color("ffb703")
const BAD := Color("ef233c")

const MOOD_COLORS := {
	"happy": Color("ffd166"),
	"excited": Color("ff9f1c"),
	"positive": Color("06d6a0"),
	"calm": Color("4cc9f0"),
	"neutral": Color("8d99ae"),
	"confused": Color("b388eb"),
	"anxious": Color("f4a261"),
	"sad": Color("577590"),
	"frustrated": Color("e76f51"),
	"angry": Color("ef233c"),
}

const FACT_COLORS := {
	"correct": Color("06d6a0"),
	"incorrect": Color("ef233c"),
	"doubtful": Color("ffb703"),
}


static func speaker_color(name: String, is_me: bool = false) -> Color:
	if is_me:
		return Color("4cc9f0")
	var h := float(abs(name.hash()) % 360) / 360.0
	return Color.from_hsv(h, 0.55, 0.92)


static func mood_color(mood) -> Color:
	if mood == null:
		return BORDER
	return MOOD_COLORS.get(str(mood), Color("8d99ae"))


static func initials(name: String) -> String:
	var parts := name.replace("_", " ").strip_edges().split(" ", false)
	if parts.is_empty():
		return "?"
	if parts.size() == 1:
		return parts[0].substr(0, 2).to_upper()
	return (parts[0].substr(0, 1) + parts[1].substr(0, 1)).to_upper()


static func duration(seconds: float) -> String:
	var s := int(round(seconds))
	if s < 60:
		return "%ds" % s
	if s < 3600:
		return "%dm %02ds" % [s / 60, s % 60]
	return "%dh %02dm" % [s / 3600, (s % 3600) / 60]


static func percent(fraction: float) -> String:
	return "%d%%" % int(round(fraction * 100.0))


static func clock(unix_time: float) -> String:
	var d := Time.get_datetime_dict_from_unix_time(int(unix_time) + _tz_offset())
	return "%02d:%02d:%02d" % [d.hour, d.minute, d.second]


static func date(unix_time: float) -> String:
	var d := Time.get_datetime_dict_from_unix_time(int(unix_time) + _tz_offset())
	return "%04d-%02d-%02d" % [d.year, d.month, d.day]


static func _tz_offset() -> int:
	return int(Time.get_time_zone_from_system().get("bias", 0)) * 60


static func panel_style(color: Color = PANEL, radius: int = 10, pad: int = 10) -> StyleBoxFlat:
	var sb := StyleBoxFlat.new()
	sb.bg_color = color
	sb.set_corner_radius_all(radius)
	sb.set_content_margin_all(pad)
	sb.border_color = BORDER
	sb.set_border_width_all(1)
	return sb
