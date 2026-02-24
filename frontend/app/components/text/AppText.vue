<script setup lang="ts">
// 1. Строго ограничиваем типы
type TextType = 'hero' | 'title' | 'body' | 'medium' | 'small' | 'base' | 'tiny' | 'large';
type TextColor = 'default' | 'muted' | 'primary' | 'error';

const props = withDefaults(
  defineProps<{
    type?: TextType;
    color?: TextColor;
    center?: boolean;
  }>(),
  {
    type: 'body',
    color: 'default',
    center: false
  }
);

// 2. Инкапсулируем дизайн внутри компонента
const typeStyles: Record<TextType, string> = {
  // text-balance делает заголовки красивыми (убирает висячие слова)
  // tracking-tight уменьшает межсимвольный интервал (модно для жирных шрифтов)
  hero: 'text-4xl md:text-6xl font-bold tracking-tight text-balance leading-[1.1]',

  large: 'text-2xl md:text-4xl font-bold tracking-tight text-pretty',

  title: 'text-xl md:text-2xl font-semibold tracking-tight',

  // leading-relaxed добавляет воздуха в большие блоки текста
  body: 'text-lg leading-relaxed text-pretty',

  base: 'text-base font-medium',

  medium: 'text-sm md:text-base font-medium',

  small: 'text-xs md:text-sm',

  tiny: 'text-[10px] leading-tight'
};

const colorStyles: Record<TextColor, string> = {
  default: 'text-foreground',
  muted: 'text-muted-foreground',
  primary: 'text-primary',
  error: 'text-destructive'
};
</script>

<template>
  <div :class="[typeStyles[type], colorStyles[color]]">
    <slot />
  </div>
</template>
