/** @type {import('tailwindcss').Config} */

const colorKeys = [
  'background',
  'foreground',
  'card',
  'card-foreground',
  'popover',
  'popover-foreground',
  'primary',
  'primary-foreground',
  'secondary',
  'secondary-foreground',
  'muted',
  'muted-foreground',
  'accent',
  'accent-foreground',
  'destructive',
  'destructive-foreground',
  'border',
  'input',
  'ring',
  'success',
  'warning',
  'error',
  'chart-1',
  'chart-2',
  'chart-3',
  'chart-4',
  'chart-5',
  'sidebar',
  'sidebar-foreground',
  'sidebar-primary',
  'sidebar-primary-foreground',
  'sidebar-accent',
  'sidebar-accent-foreground',
  'sidebar-border',
  'sidebar-ring',
  'neon-purple'
];

module.exports = {
  // Указываем, где искать классы (для Nuxt 4 это папка app)
  content: ['./app/**/*.{vue,js,ts,jsx,tsx}', './app.vue'],
  theme: {
    extend: {
      colors: {
        ...Object.fromEntries(colorKeys.map((key) => [key, `rgb(var(--${key}) / <alpha-value>)`]))
      }
    }
  },
  plugins: []
};
