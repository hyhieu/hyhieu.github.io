return {
  "nvim-tree/nvim-tree.lua",
  lazy = false,
  dependencies = {
    "nvim-tree/nvim-web-devicons",
  },
  config = function()
    vim.g.loaded = 1
    vim.g.loaded_netrwPlugin = 1
    
    require("nvim-tree").setup({
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

    vim.keymap.set("n", "<leader>e", ":NvimTreeToggle<CR>", {})
    vim.keymap.set("n", "<leader>f", ":NvimTreeFocus<CR>", {})
  end,
}

