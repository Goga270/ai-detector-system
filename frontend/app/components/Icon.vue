<script setup lang="ts">
import type { IconSize, IconUi } from '~/types';

const props = withDefaults(
  defineProps<{
    iconName: string;
    ui?: IconUi;
    size?: IconSize;
  }>(),
  {
    ui: 'primary',
    size: 'm'
  }
);

const uiClasses = {
  primary: 'icon-primary',
  secondary: 'icon-secondary',
  dark: 'icon-dark'
};

const sizeClasses = {
  s: 'icon-s',
  m: 'icon-m',
  l: 'icon-l'
};

const classes = computed(() => {
  return [uiClasses[props.ui], sizeClasses[props.size]];
});
</script>

<template>
  <div :class="classes">{{ iconName }}</div>
</template>
