<script setup lang="ts">
import { useStore } from 'vuex';

import Btn from '~/components/buttons/Btn.vue';
import CustomIcon from '~/components/icons/CustomIcon.vue';
import AppText from '~/components/text/AppText.vue';
import type { RootState } from '~/store';

const props = withDefaults(
  defineProps<{
    modelValue: string;
    placeholder?: string;
    label?: string;
    minHeight?: string;
    error?: string;
  }>(),
  {
    placeholder: 'Введите текст для анализа...',
    label: '',
    minHeight: '400px',
    error: ''
  }
);

const emit = defineEmits<{
  (event: 'update:modelValue', value: string): void;
  (event: 'analyze'): void;
  (event: 'clear'): void;
}>();

const store = useStore<RootState>();

const charCount = computed(() => props.modelValue.length);

const wordCount = computed(() => {
  if (!props.modelValue.trim()) {
    return 0;
  }

  return props.modelValue.trim().split(/\s+/).length;
});

const analyzeIsLoading = computed(() => !!store.state.detector?.loading);

function handleChange(event: Event) {
  emit('update:modelValue', event.target?.value);
}

function handleAnalyze() {
  emit('analyze');
}

function handleClear() {
  emit('clear');
}
</script>

<template>
  <div class="relative">
    <label v-if="label" class="text-sm font-medium text-muted-foreground ml-1">
      {{ label }}
    </label>
    <div class="glass-card rounded-2xl overflow-hidden hover:border-primary/50 group">
      <textarea
        class="overflow-hidden flex outline-none focus:outline-none custom-scrollbar w-full min-h-[400px] resize-none bg-input/30 p-6 text-foreground placeholder:text-muted-foreground text-base leading-relaxed"
        :value="modelValue"
        placeholder="Вставьте или введите текст для проверки на наличие ИИ-генерации..."
        @input="handleChange"
      />
      <div
        class="flex items-center justify-between border border-x-0 border-b-0 border-border/15 bg-card/60 px-6 py-4 group-hover:border-primary/50 transition-all duration-300"
      >
        <div class="flex flex-col sm:flex-row items-start sm:items-center gap-6">
          <div class="flex items-center gap-2 text-sm text-muted-foreground">
            <Icon name="lucide:align-left"></Icon>
            <AppText type="small" color="muted">Слов:</AppText>
            <AppText type="small" weight="medium">{{ wordCount }}</AppText>
          </div>
          <div class="flex items-center gap-2 text-sm text-muted-foreground">
            <Icon name="lucide:hash" />
            <AppText type="small" color="muted">Символов:</AppText>
            <AppText type="small" weight="medium">{{ charCount }}</AppText>
          </div>
        </div>
        <div class="flex sm-flex items-center gap-3">
          <Btn
            class="hidden sm:inline-flex"
            text="Очистить"
            ui="secondary"
            :disabled="!modelValue"
            :loading="analyzeIsLoading"
            @click="handleClear"
          />
          <Btn class="cursor-pointer sm:hidden text-destructive" ui="secondary" is-icon @click="handleClear">
            <Icon name="lucide:trash-2" />
          </Btn>
          <Btn
            text="Проверить"
            :disabled="!modelValue"
            :loading="analyzeIsLoading"
            loading-text="Анализируем"
            @click="handleAnalyze"
          >
            <template #iconBefore>
              <Icon v-show="analyzeIsLoading" name="lucide:loader-2 animate-spin" />
            </template>
          </Btn>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped></style>
