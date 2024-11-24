return {
  "nvim-treesitter/nvim-treesitter",
  build=":TSUpdate",
  config = function()
    require("nvim-treesitter.configs").setup({
      ensure_installed = {
        "c",
        "cmake",
        "cpp",
        "cuda",
        "lua",
        "markdown",
        "markdown_inline",
        "python", 
        "toml",
      },
      highlight = { enable = true },
      indent = { enable = true },
    })
  end,
}

