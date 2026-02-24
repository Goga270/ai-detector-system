<script setup lang="ts">
import type { BtnSize, BtnUi } from '~/types';

const props = withDefaults(
  defineProps<{
    text?: string;
    ui?: BtnUi;
    size?: BtnSize;
    loading?: boolean;
    loadingText?: string;
    disabled?: boolean;
    isIcon?: boolean;
  }>(),
  {
    loading: false,
    disabled: false,
    size: 'm',
    ui: 'primary',
    isIcon: false
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
  dark: 'btn-dark',
  ghost: 'btn-ghost'
};

const sizeClasses = {
  s: 'btn-s',
  m: 'btn-m',
  l: 'btn-l'
};

const iconSizeClasses = {
  s: 'icon-s',
  m: 'icon-m',
  l: 'icon-l'
};

const classes = computed(() => {
  const baseClasses = [uiClasses[props.ui], sizeClasses[props.size]];

  if (!props.loading && !props.disabled && props.ui === 'primary') {
    baseClasses.push('glow-button');
  }

  if (props.isIcon) {
    baseClasses.push(...[iconSizeClasses[props.size], 'justify-center']);
  }

  return baseClasses;
});
</script>

<template>
  <button :disabled="disabled || loading" :class="classes" @click="handleClick">
    <slot name="iconAfter" />
    <slot>
      {{ loading ? loadingText || text : text }}
    </slot>
    <slot name="iconBefore" />
  </button>
</template>

<style scoped></style>
