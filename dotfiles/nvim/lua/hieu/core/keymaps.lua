vim.g.mapleader = " "

local keymap = vim.keymap

keymap.set("n", "j", "gj", { noremap = true })
keymap.set("n", "k", "gk", { noremap = true })

keymap.set("n", "x", '"_x')

-- nvim-tree
-- keymap.set("n", "<leader>e", ":NvimTreeToggle<CR>")
keymap.set("n", "<leader>e", ":NvimTreeFocus<CR>")

