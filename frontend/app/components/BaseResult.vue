<script setup lang="ts">
import GaugeMeter from '~/components/graphics/GaugeMeter.vue';
import AppText from '~/components/text/AppText.vue';
import type { DetectionResult } from '~/store/modules/detector';

const props = defineProps<{
  result?: DetectionResult | null;
  loading: boolean;
}>();

const verdict = computed(() => {
  const prob = props.result?.analysis.ai_probability;

  if (!prob) {
    return {
      icon: ''
    };
  }

  if (prob > 75) {
    return {
      text: 'Вероятно, ИИ',
      icon: 'lucide:bot',
      color: '!text-destructive',
      bg: 'bg-destructive/10',
      border: 'border-destructive/20',
      glow: 'shadow-destructive/20'
    };
  }
  if (prob > 35) {
    return {
      text: 'Смешанный текст',
      icon: 'lucide:alert-circle',
      color: '!text-warning',
      bg: 'bg-warning/10',
      border: 'border-warning/20',
      glow: 'shadow-warning/20'
    };
  }

  return {
    text: 'Написано человеком',
    icon: 'lucide:user-check',
    color: '!text-success',
    bg: 'bg-success/10',
    border: 'border-success/20',
    glow: 'shadow-success/20'
  };
});

/**
 * Перевод уверенности на русский
 */
const confidenceRU = computed(() => {
  const conf = props.result?.analysis.confidence;

  if (!conf) {
    return '';
  }

  const map = { High: 'Высокая', Medium: 'Средняя', Low: 'Низкая' };

  return map[conf] || conf;
});
</script>

<template>
  <div
    class="mt-12 rounded-3xl border border-border/10 bg-card/40 p-6 shadow-2xl backdrop-blur-xl md:p-10 min-h-[400px] flex flex-col justify-center"
  >
    <Transition name="fade-slow" mode="out-in">
      <div v-if="loading" class="flex flex-col items-center justify-center py-12 text-center">
        <div class="relative mb-6">
          <div class="h-24 w-24 animate-spin rounded-full border-4 border-primary/10 border-t-primary" />
          <div class="absolute inset-0 flex items-center justify-center">
            <Icon name="lucide:bot" class="w-10 h-10 text-primary animate-pulse" />
          </div>
        </div>
        <AppText type="title" class="mb-2">Анализируем контент...</AppText>
        <AppText type="small" color="muted">Обычно это занимает несколько секунд</AppText>
      </div>

      <div v-else-if="result" class="w-full">
        <div class="flex items-center justify-center gap-3 mb-6">
          <div class="h-px w-12 bg-gradient-to-r from-transparent to-border/50" />
          <AppText type="title" class="uppercase tracking-[0.2em] opacity-80">Результаты анализа</AppText>
          <div class="h-px w-12 bg-gradient-to-l from-transparent to-border/50" />
        </div>

        <div class="space-y-10 mb-6">
          <GaugeMeter :value="result.analysis.ai_probability" />

          <div v-if="verdict" class="flex justify-center">
            <div
              :class="[verdict.bg, verdict.border, verdict.glow]"
              class="inline-flex items-center gap-2 rounded-full border px-6 py-2 shadow-lg transition-all duration-500"
            >
              <Icon :name="verdict.icon" :class="verdict.color" class="w-6 h-6" />
              <AppText :class="verdict.color" type="medium">
                {{ verdict.text }}
              </AppText>
            </div>
          </div>
        </div>

        <div class="grid grid-cols-1 gap-5 sm:grid-cols-3">
          <div
            class="group relative overflow-hidden rounded-2xl border border-white/5 bg-secondary/20 p-6 transition-all hover:border-primary/30 hover:bg-secondary/30"
          >
            <div
              class="absolute -right-4 -top-4 opacity-5 transition-transform group-hover:scale-110 group-hover:opacity-10"
            >
              <Icon name="lucide:languages" class="w-24 h-24 text-primary" />
            </div>
            <div class="relative z-10 flex flex-col items-center text-center">
              <Icon name="lucide:languages" class="mb-3 w-6 h-6 text-primary/70" />
              <AppText type="small" color="muted" class="mb-1 uppercase tracking-widest font-bold">Лексика</AppText>
              <AppText type="large" class="mb-2 tabular-nums">
                {{ (result.analysis.features.unique_word_ratio * 100).toFixed(1) }}%
              </AppText>
              <AppText type="small" color="muted" class="leading-tight opacity-60">Уникальность слов</AppText>
            </div>
          </div>
          <div
            class="group relative overflow-hidden rounded-2xl border border-white/5 bg-secondary/20 p-6 transition-all hover:border-accent/30 hover:bg-secondary/30"
          >
            <div
              class="absolute -right-4 -top-4 opacity-5 transition-transform group-hover:scale-110 group-hover:opacity-10"
            >
              <Icon name="lucide:target" class="w-24 h-24 text-accent" />
            </div>
            <div class="relative z-10 flex flex-col items-center text-center">
              <Icon name="lucide:target" class="mb-3 w-6 h-6 text-accent/70" />
              <AppText type="small" color="muted" class="mb-1 uppercase tracking-widest font-bold">Точность</AppText>
              <AppText type="large" class="mb-2">{{ confidenceRU }}</AppText>
              <AppText type="small" color="muted" class="leading-tight opacity-60">Доверие алгоритма</AppText>
            </div>
          </div>
          <div
            class="group relative overflow-hidden rounded-2xl border border-white/5 bg-secondary/20 p-6 transition-all hover:border-success/30 hover:bg-secondary/30"
          >
            <div
              class="absolute -right-4 -top-4 opacity-5 transition-transform group-hover:scale-110 group-hover:opacity-10"
            >
              <Icon name="lucide:feather" class="w-24 h-24 text-success" />
            </div>
            <div class="relative z-10 flex flex-col items-center text-center">
              <Icon name="lucide:feather" class="mb-3 w-6 h-6 text-success/70" />
              <AppText type="small" color="muted" class="mb-1 uppercase tracking-widest font-bold">Стиль</AppText>
              <AppText type="large" class="mb-2">
                {{ result.analysis.features.stopword_ratio > 0.4 ? 'Сложный' : 'Простой' }}
              </AppText>
              <AppText type="small" color="muted" class="leading-tight opacity-60">Структура текста</AppText>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.fade-slow-enter-active,
.fade-slow-leave-active {
  transition:
    opacity 0.5s ease,
    transform 0.5s ease;
}
.fade-slow-enter-from,
.fade-slow-leave-to {
  opacity: 0;
  transform: scale(0.98);
}
</style>
