-- Line numbers
vim.opt.number = true

-- Leader key
vim.g.mapleader = " "
vim.g.background = "dark"

-- Tabs & indentations
vim.opt.tabstop = 2
vim.opt.shiftwidth = 2
vim.opt.expandtab = true
vim.opt.autoindent = true

-- Line wrap
vim.opt.wrap = false

-- Search case sensitivity
vim.opt.ignorecase = true
vim.opt.smartcase = true

-- Appearances
vim.opt.termguicolors = true
vim.opt.background = "dark"
-- vim.opt.signcolumn = "yes"

-- Clipboard
vim.opt.clipboard:append("unnamedplus")

-- Split panes
vim.opt.splitbelow = true
vim.opt.splitright = true

-- Code folding
vim.opt.foldopen = "search"
vim.opt.foldmethod = "expr"
vim.opt.foldenable = false
vim.opt.foldexpr = "v:lua.vim.treesitter.foldexpr()"

-- Code folding
vim.opt.autochdir = true

-- Keymaps
vim.keymap.set("n", "j", "gj", { noremap = true })
vim.keymap.set("n", "k", "gk", { noremap = true })
vim.keymap.set("n", "x", '"_x', { noremap = true })


