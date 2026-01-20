<script setup lang="ts">
import type { BtnSize, BtnUi } from '~/types';

const props = withDefaults(
  defineProps<{
    text: string;
    ui?: BtnUi;
    size?: BtnSize;
    loading?: boolean;
    loadingText?: string;
    disabled?: boolean;
  }>(),
  {
    loading: false,
    disabled: false,
    size: 'm',
    ui: 'primary'
  }
);

const emit = defineEmits<{
  (event: 'click'): void;
}>();

function handleClick() {
  emit('click');
}

const uiClasses = {
  primary: 'btn-primary',
  secondary: 'btn-secondary',
  link: 'btn-link',
  dark: 'btn-dark'
};

const sizeClasses = {
  s: 'btn-s',
  m: 'btn-m',
  l: 'btn-l'
};

const classes = computed(() => {
  const baseClasses = [uiClasses[props.ui], sizeClasses[props.size]];

  if (!props.loading && !props.disabled && props.ui === 'primary') {
    baseClasses.push('glow-button');
  }

  return baseClasses;
});
</script>

<template>
  <button :disabled="disabled || loading" :class="classes" @click="handleClick">
    {{ loading ? loadingText || text : text }}
  </button>
</template>

<style scoped></style>
