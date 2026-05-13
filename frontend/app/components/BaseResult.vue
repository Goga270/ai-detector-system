<script setup lang="ts">
import { useStore } from 'vuex';

import Btn from '~/components/buttons/Btn.vue';
import GaugeMeter from '~/components/graphics/GaugeMeter.vue';
import AppText from '~/components/text/AppText.vue';
import type { RootState } from '~/store';
import type { DetectionResult } from '~/types';

const props = defineProps<{
  result?: DetectionResult | null;
  loading: boolean;
}>();

const store = useStore<RootState>();

const verdict = computed(() => {
  const prob = props.result?.aiPercentage;

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
 * Логика нарезки по спанам
 */
const highlightedText = computed(() => {
  const originalText = store.state.detector?.currentText || '';

  const spans = props.result?.spans || [];

  if (!spans.length || !originalText) {
    return [{ text: originalText, type: 'NONE', prob: 0 }];
  }

  const sortedSpans = [...spans].sort((a, b) => a.startChar - b.startChar);

  const parts = [];

  let lastIndex = 0;

  for (const span of sortedSpans) {
    if (span.startChar > lastIndex) {
      parts.push({
        text: originalText.slice(lastIndex, span.startChar),
        type: 'NONE',
        prob: 0
      });
    }

    const isAI = span.avgConfidence > 0.5;

    parts.push({
      text: originalText.slice(span.startChar, span.endChar),
      type: isAI ? 'AI' : 'HUMAN',
      prob: span.avgConfidence
    });

    lastIndex = span.endChar;
  }

  if (lastIndex < originalText.length) {
    parts.push({
      text: originalText.slice(lastIndex),
      type: 'NONE',
      prob: 0
    });
  }

  return parts;
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
          <GaugeMeter :value="result.aiPercentage" />

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
              <Icon name="lucide:target" class="w-24 h-24 text-primary" />
            </div>
            <div class="relative z-10 flex flex-col items-center text-center">
              <Icon name="lucide:target" class="mb-3 w-6 h-6 text-primary/70" />
              <AppText type="small" color="muted" class="mb-1 uppercase tracking-widest font-bold">Уверенность</AppText>
              <AppText type="large" class="mb-2 tabular-nums"> {{ (result.confidence * 100).toFixed(1) }}% </AppText>
              <AppText type="small" color="muted" class="leading-tight opacity-60">Уверенность нейросети</AppText>
            </div>
          </div>
          <div
            class="group relative overflow-hidden rounded-2xl border border-white/5 bg-secondary/20 p-6 transition-all hover:border-accent/30 hover:bg-secondary/30"
          >
            <div
              class="absolute -right-4 -top-4 opacity-5 transition-transform group-hover:scale-110 group-hover:opacity-10"
            >
              <Icon name="lucide:network" class="w-24 h-24 text-accent" />
            </div>
            <div class="relative z-10 flex flex-col items-center text-center">
              <Icon name="lucide:network" class="mb-3 w-6 h-6 text-accent/70" />
              <AppText type="small" color="muted" class="mb-1 uppercase tracking-widest font-bold"
                >Согласие алгоритмов</AppText
              >
              <AppText type="large" class="mb-2">{{ (result.judgeAgreement * 100).toFixed(1) }}%</AppText>
              <AppText type="small" color="muted" class="leading-tight opacity-60">Доверие алгоритма</AppText>
            </div>
          </div>
          <div
            :class="`group relative overflow-hidden rounded-2xl border border-white/5 bg-secondary/20 p-6 transition-all ${result.needsHumanReview ? 'hover:border-warning/30' : 'hover:border-success/30'} hover:bg-secondary/30`"
          >
            <div
              class="absolute -right-4 -top-4 opacity-5 transition-transform group-hover:scale-110 group-hover:opacity-10"
            >
              <Icon
                :name="result.needsHumanReview ? 'lucide:user-search' : 'lucide:bot-message-square'"
                :class="['w-24', 'h-24', result.needsHumanReview ? 'text-warning' : 'text-success']"
              />
            </div>
            <div class="relative z-10 flex flex-col items-center text-center">
              <Icon
                :name="result.needsHumanReview ? 'lucide:user-search' : 'lucide:bot-message-square'"
                :class="['mb-3', 'w-6', 'h-6', result.needsHumanReview ? 'text-warning/70' : 'text-success/70']"
              />
              <AppText type="small" color="muted" class="mb-1 uppercase tracking-widest font-bold">Резолюция</AppText>
              <AppText type="large" :class="['mb-2', result.needsHumanReview ? 'text-warning' : 'text-success']">
                {{ result.needsHumanReview ? 'Спорно' : 'Надежно' }}
              </AppText>
              <AppText type="small" color="muted" class="leading-tight opacity-60">{{
                result.reviewReason || 'Ручная проверка не требуется'
              }}</AppText>
            </div>
          </div>
        </div>

        <div class="mt-10 flex justify-center">
          <UiDialog>
            <UiDialogTrigger as-child>
              <Btn
                text="Подробный разбор текста"
                ui="secondary"
                size="l"
                class="hover:border-primary/50 hover:bg-primary/10 transition-colors"
              >
                <template #iconBefore>
                  <Icon name="lucide:file-search" class="text-primary" />
                </template>
              </Btn>
            </UiDialogTrigger>

            <UiDialogContent
              class="sm:max-w-[700px] max-h-[85vh] bg-background/95 backdrop-blur-xl border-border/20 shadow-2xl flex flex-col"
            >
              <UiDialogHeader>
                <UiDialogTitle class="text-2xl font-bold flex items-center gap-2">
                  <Icon name="lucide:microscope" class="text-primary w-6 h-6" />
                  Анализ предложений
                </UiDialogTitle>
                <UiDialogDescription class="text-muted-foreground">
                  Цветом выделены фрагменты текста, которые алгоритм распознал как созданные ИИ или написанные
                  человеком.
                </UiDialogDescription>
              </UiDialogHeader>

              <!-- Обоснование от ML-модели (Explanation) -->
              <div
                v-if="result.explanation"
                class="p-4 rounded-xl bg-secondary/30 border border-white/5 mb-4 text-sm text-muted-foreground leading-relaxed"
              >
                <span class="font-bold text-foreground">Заключение системы:</span> {{ result.explanation }}
              </div>

              <!-- Текст с подсветкой -->
              <div class="flex-1 overflow-y-auto custom-scrollbar p-4 bg-card/40 rounded-xl border border-border/10">
                <p class="text-base leading-loose text-foreground/90 whitespace-pre-wrap">
                  <template v-for="(part, index) in highlightedText" :key="index">
                    <!-- Если это ИИ -->
                    <span
                      v-if="part.type === 'AI'"
                      class="bg-destructive/20 text-destructive-foreground px-1 rounded border border-destructive/30 shadow-[0_0_10px_rgba(var(--destructive),0.2)]"
                      :title="`Вероятность ИИ: ${(part.prob * 100).toFixed(1)}%`"
                    >
                      {{ part.text }}
                    </span>

                    <!-- Если это Человек -->
                    <span
                      v-else-if="part.type === 'HUMAN'"
                      class="bg-success/20 text-success-foreground px-1 rounded border border-success/30"
                      :title="`Вероятность Человека: ${(part.prob * 100).toFixed(1)}%`"
                    >
                      {{ part.text }}
                    </span>

                    <!-- Обычный текст (нейтральный) -->
                    <span v-else>{{ part.text }}</span>
                  </template>
                </p>
              </div>

              <!-- Легенда внизу модалки -->
              <div class="mt-4 flex items-center gap-4 text-xs font-medium text-muted-foreground">
                <div class="flex items-center gap-1.5">
                  <div class="w-3 h-3 rounded bg-destructive/20 border border-destructive/30"></div>
                  Сгенерировано ИИ
                </div>
                <div class="flex items-center gap-1.5">
                  <div class="w-3 h-3 rounded bg-success/20 border border-success/30"></div>
                  Написано человеком
                </div>
              </div>
            </UiDialogContent>
          </UiDialog>
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
