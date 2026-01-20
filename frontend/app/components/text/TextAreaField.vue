<script setup lang="ts">
import { useStore } from 'vuex';

import Btn from '~/components/buttons/Btn.vue';
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
  emit('update:modelValue', event.target.value);
}

function handleAnalyze() {
  emit('analyze');
}
</script>

<template>
  <div class="relative">
    <label v-if="label" class="text-sm font-medium text-muted-foreground ml-1">
      {{ label }}
    </label>
    <div class="glass-card rounded-2xl overflow-hidden hover:border-primary/50 group">
      <textarea
        class="overflow-hidden flex outline-none focus:outline-none custom-scrollbar w-full min-h-[400px] resize-none bg-[#0B0E14] p-6 text-foreground placeholder:text-muted-foreground text-base leading-relaxed"
        :value="modelValue"
        placeholder="Paste or type your content here to analyze for AI-generated text..."
        @input="handleChange"
      />
      <div
        class="flex items-center justify-between border border-x-0 border-b-0 border-border bg-card px-6 py-4 group-hover:border-primary/50 transition-all duration-300"
      >
        <div class="flex flex-col sm:flex-row items-start sm:items-center gap-6">
          <div class="flex items-center gap-2 text-sm text-muted-foreground">
            <div class="h-4 w-4">Ic</div>
            <span>Words:</span>
            <span class="font-medium text-foreground">{{ wordCount }}</span>
          </div>
          <div class="flex items-center gap-2 text-sm text-muted-foreground">
            <div class="h-4 w-4">2</div>
            <span>Characters:</span>
            <span class="font-medium text-foreground">{{ charCount }}</span>
          </div>
        </div>
        <div class="flex sm-flex items-center gap-3">
          <Btn
            class="hidden sm:inline-flex"
            text="Cancel"
            ui="secondary"
            :disabled="!modelValue"
            :loading="analyzeIsLoading"
          />
          <Icon class="cursor-pointer sm:hidden" ui="secondary" icon-name="O" />
          <Btn
            text="Analyze Now"
            :disabled="!modelValue"
            :loading="analyzeIsLoading"
            loading-text="Analyzing..."
            @click="handleAnalyze"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped></style>
