// @ts-check
import prettierConfig from 'eslint-config-prettier';
import prettierPlugin from 'eslint-plugin-prettier';

import withNuxt from './.nuxt/eslint.config.mjs';

export default withNuxt(
  {
    // Настройки для всех файлов проекта
    files: ['**/*.ts', '**/*.vue'],
    plugins: {
      prettier: prettierPlugin
    },
    rules: {
      'prettier/prettier': 'error',

      'no-console': process.env.NODE_ENV === 'production' ? 'error' : 'warn',
      'no-debugger': 'error',
      curly: 'error',

      // TypeScript
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/consistent-type-imports': ['error', { prefer: 'type-imports' }],

      // Stylistic (оставляем только те, что не мешают Prettier)
      '@stylistic/padding-line-between-statements': [
        'error',
        { blankLine: 'always', prev: ['const', 'let'], next: '*' },
        { blankLine: 'always', prev: '*', next: 'return' }
      ],

      // Vue
      'vue/multi-word-component-names': 'off',
      'vue/component-name-in-template-casing': ['error', 'PascalCase']
    }
  },
  prettierConfig
);
