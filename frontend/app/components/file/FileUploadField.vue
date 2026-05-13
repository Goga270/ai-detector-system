<script setup lang="ts">
import { ref } from 'vue';

import Btn from '~/components/buttons/Btn.vue';
import AppText from '~/components/text/AppText.vue';
import type { UploadedFile } from '~/types';

const emit = defineEmits(['analyze']);

const currentIdCounter = ref<number>(0);

const isDragging = ref(false);

const files = ref<UploadedFile[]>([]);

const fileInput = ref<HTMLInputElement | null>(null);

const handleFiles = (newFiles: FileList | null) => {
  if (!newFiles) return;

  files.value = [];

  Array.from(newFiles).forEach((file) => {
    const fileObj: UploadedFile = {
      id: currentIdCounter.value,
      file,
      name: file.name,
      size: file.size,
      progress: 0,
      status: 'loading'
    };

    currentIdCounter.value = currentIdCounter.value + 1;

    files.value.push(fileObj);

    const bytesPerSecond = 2 * 1024 * 1024;

    const baseDuration = Math.max(400, (file.size / bytesPerSecond) * 1000);

    const tickTime = 50;

    let elapsed = 0;

    const interval = setInterval(() => {
      const index = files.value.findIndex((f) => f.id === fileObj.id);

      if (index === -1 || !files.value[index]) {
        clearInterval(interval);

        return;
      }

      elapsed += tickTime;

      const jitter = Math.random() * 3; // случайное отклонение 0-3%

      const calculatedProgress = Math.floor((elapsed / baseDuration) * 100);

      const visibleProgress = calculatedProgress + jitter;

      files.value[index].progress = Math.floor(visibleProgress);

      // Завершение
      if (files.value[index].progress >= 100) {
        files.value[index].progress = 100;
        files.value[index].status = 'ready';
        clearInterval(interval);
      }
    }, tickTime);
  });
};

const onDrop = (e: DragEvent) => {
  isDragging.value = false;
  handleFiles(e.dataTransfer?.files || null);
};

const removeFile = (id: number) => {
  files.value = files.value.filter((f) => f.id !== id);
};

const formatSize = (bytes: number) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;

  const sizes = ['Bytes', 'KB', 'MB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};
</script>

<template>
  <div class="space-y-4">
    <!-- ЗОНА DROPZONE -->
    <div
      class="relative group cursor-pointer transition-all duration-300"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop.prevent="onDrop"
      @click="fileInput?.click()"
    >
      <input
        ref="fileInput"
        type="file"
        class="hidden"
        accept=".pdf"
        @change="handleFiles(($event.target as HTMLInputElement).files)"
      />

      <div
        :class="[
          'flex flex-col items-center justify-center p-12 rounded-[32px] border-2 border-dashed transition-all duration-300',
          isDragging
            ? 'border-primary bg-primary/10 scale-[0.99]'
            : 'border-border/20 bg-input/20 hover:border-primary/40'
        ]"
      >
        <div
          class="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-6 shadow-xl shadow-primary/5"
        >
          <Icon name="lucide:file-up" class="w-8 h-8 text-primary" />
        </div>

        <AppText type="body" class="mb-2">Перетащите файлы для анализа</AppText>
        <AppText type="small" color="muted">PDF до 10MB</AppText>

        <Btn text="Выбрать файлы" ui="secondary" size="m" class="mt-6 pointer-events-none" />
      </div>
    </div>

    <div v-if="files.length > 0" class="space-y-3">
      <TransitionGroup name="list">
        <div
          v-for="file in files"
          :key="file.id"
          class="glass-card !rounded-2xl p-4 flex items-center gap-4 border-border/10 bg-secondary/20"
        >
          <!-- Иконка типа -->
          <div class="w-10 h-10 rounded-lg bg-background flex items-center justify-center border border-border/10">
            <Icon
              :name="file.file.name.endsWith('.pdf') ? 'lucide:file-text' : 'lucide:file-code'"
              class="w-5 h-5 text-primary"
            />
          </div>

          <!-- Инфо -->
          <div class="flex-1 min-w-0">
            <div class="flex justify-between items-start mb-1">
              <div class="truncate pr-4">
                <AppText type="small" class="font-bold truncate">{{ file.file.name }}</AppText>
                <AppText type="tiny" color="muted">{{ formatSize(file.file.size) }}</AppText>
              </div>

              <div class="flex items-center gap-2">
                <div v-if="file.status === 'ready'" class="flex items-center gap-1.5 text-success">
                  <Icon name="lucide:check-circle" class="w-4 h-4" />
                  <span class="text-[10px] font-bold uppercase">Готов</span>
                </div>
                <button
                  class="text-muted-foreground hover:text-destructive transition-colors"
                  @click.stop="removeFile(file.id)"
                >
                  <Icon name="lucide:x" class="w-4 h-4" />
                </button>
              </div>
            </div>

            <!-- Прогресс бар из Shadcn -->
            <UiProgress v-if="file.status === 'loading'" :model-value="file.progress" class="h-1 bg-primary/10" />
          </div>
        </div>
      </TransitionGroup>

      <!-- КНОПКА АНАЛИЗА ФАЙЛОВ -->
      <div class="flex justify-end pt-4">
        <Btn
          :text="`Анализировать файл`"
          size="l"
          class="btn-primary glow-button"
          :disabled="files.some((f) => f.status === 'loading')"
          @click="$emit('analyze', files)"
        />
      </div>
    </div>

    <AppText type="tiny" color="muted" center class="mt-4 text-center">
      Поддерживаемые форматы: PDF. Максимальный размер: 10MB.
    </AppText>
  </div>
</template>

<style scoped>
.list-enter-active,
.list-leave-active {
  transition: all 0.3s ease;
}
.list-enter-from,
.list-leave-to {
  opacity: 0;
  transform: translateX(20px);
}
</style>
