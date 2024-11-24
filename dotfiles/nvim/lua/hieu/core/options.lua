local opt = vim.opt

-- Line numbers
opt.number = true

-- Tabs & indentations
opt.tabstop = 2
opt.shiftwidth = 2
opt.expandtab = true
opt.autoindent = true

-- Line wrap
opt.wrap = false

-- Search case sensitivity
opt.ignorecase = true
opt.smartcase = true

-- Appearances
opt.termguicolors = true
opt.background = "dark"
-- opt.signcolumn = "yes"

-- Clipboard
opt.clipboard:append("unnamedplus")

-- Split panes
opt.splitbelow = true
opt.splitright = true

-- Code folding
opt.foldmethod = "indent"
opt.foldenable = false
opt.foldlevel = 2
opt.foldtext = "CustomFoldText()"
opt.foldopen = "search"

-- Code folding
opt.autochdir = true

