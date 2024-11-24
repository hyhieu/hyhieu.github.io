local setup, nvimtree = pcall(require, "nvim-tree")
if not setup then
  return
end

vim.g.loaded = 1
vim.g.loaded_netrwPlugin = 1

nvimtree.setup({
  sort = {
    sorter = "case_sensitive",
  },
  filters = {
    dotfiles = true,
  },
  -- disable window picker so it works with splits
  actions = {
    open_file = {
      window_picker = {
        enable = false,
      },
    },
  },
})

