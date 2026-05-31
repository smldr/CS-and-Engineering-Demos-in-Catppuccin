import matplotlib as mpl
import catppuccin
from catppuccin.extras.matplotlib import load_color

_p = catppuccin.PALETTE.mocha.identifier
mpl.style.use(_p)

teal     = load_color(_p, "teal")
peach    = load_color(_p, "peach")
green    = load_color(_p, "green")
lavender = load_color(_p, "lavender")
flamingo = load_color(_p, "flamingo")
sky      = load_color(_p, "sky")
mauve    = load_color(_p, "mauve")
red      = load_color(_p, "red")
yellow   = load_color(_p, "yellow")
sapphire = load_color(_p, "sapphire")
maroon   = load_color(_p, "maroon")
overlay0 = load_color(_p, "overlay0")
surface0 = load_color(_p, "surface0")
surface1 = load_color(_p, "surface1")
base     = load_color(_p, "base")
text_col = load_color(_p, "text")

PANE = (*mpl.colors.to_rgb(base), 0.9)
