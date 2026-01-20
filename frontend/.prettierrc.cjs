/** @type {import("prettier").Config} */
module.exports = {
  printWidth: 120,
  tabWidth: 2,
  trailingComma: 'none',
  singleQuote: true,
  semi: true,
  plugins: ['@trivago/prettier-plugin-sort-imports'],
  importOrder: ['^vue', '^nuxt', '<THIRD_PARTY_MODULES>', '^#', '^~/(.*)$', '^@/(.*)$', '^[./]'],
  importOrderSeparation: true,
  importOrderSortSpecifiers: true,
  importOrderParserPlugins: ['typescript', 'decorators-legacy', 'jsx']
};
