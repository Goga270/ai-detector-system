<script setup lang="ts">
import { ref, watch } from 'vue';
import { useStore } from 'vuex';

import AppBadge from '~/components/AppBadge.vue';
import FileUploadField from '~/components/file/FileUploadField.vue';
import CustomIcon from '~/components/icons/CustomIcon.vue';
import AppText from '~/components/text/AppText.vue';
import type { RootState } from '~/store';
import type { UploadedFile } from '~/types';

useHead({
  title: 'Главная',
  meta: [{ name: 'description', content: 'Главная страница' }]
});

const typeFormat = ref('text');

const features = [
  {
    id: 1,
    title: 'Мгновенный анализ',
    subTitle: 'Результат за секунды',
    iconName: 'lucide:zap'
  },
  {
    id: 2,
    title: 'Точность 99%',
    subTitle: 'Надежное распознавание',
    iconName: 'lucide:shield-check'
  },
  {
    id: 3,
    title: 'Любой объем текста',
    subTitle: 'Без ограничений по длине',
    iconName: 'lucide:infinity'
  }
];

const store = useStore<RootState>();

const resultRef = ref<HTMLElement | null>(null);

const topRef = ref<HTMLElement | null>(null);

const analyzeText = computed({
  get: () => store.state.detector?.currentText || '',
  set: (value: string) => {
    store.commit('detector/SET_TEXT', value);
  }
});

const analyzeResult = computed(() => {
  return store.state.detector?.result;
});

const analyzeIsLoading = computed(() => !!store.state.detector?.loading);

async function handleAnalyze() {
  store.dispatch('detector/analyzeText');

  await nextTick(() => {});

  if (resultRef.value) {
    const element = resultRef.value;

    element.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });
  }
}

const handleClear = async () => {
  store.commit('detector/SET_TEXT', '');
  store.commit('detector/SET_RESULT', null);

  if (topRef.value) {
    const element = topRef.value;

    element.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });
  }
};

const handleAnalyzeFile = (files: UploadedFile[]) => {
  const fileToUpload = files[0]?.file;

  if (fileToUpload) {
    store.dispatch('detector/analyzeFile', fileToUpload);
  }
};

watch(typeFormat, () => {
  handleClear();
});
</script>

<template>
  <div ref="topRef" class="mx-auto max-w-4xl relative">
    <div class="mb-12 text-center">
      <AppBadge class="mb-4" text="AI-Powered Detection" />
      <AppText type="hero" class="mb-4">AI Content <span class="text-primary">Detector</span></AppText>
      <AppText type="body" color="muted" class="mx-auto max-w-2xl">
        Проанализируйте любой текст, чтобы определить, написан ли он человеком или создан нейросетью. Получите точный
        результат за считанные секунды с помощью наших передовых алгоритмов.
      </AppText>
    </div>
    <div class="mb-10 grid grid-cols-1 gap-4 sm:grid-cols-3">
      <div
        v-for="feature in features"
        :key="feature.id"
        class="flex items-center gap-3 rounded-lg border border-border/15 bg-card/60 p-4"
      >
        <CustomIcon :icon-name="feature.iconName" ui="dark" />
        <div>
          <AppText type="medium">{{ feature.title }}</AppText>
          <AppText type="small" color="muted">{{ feature.subTitle }}</AppText>
        </div>
      </div>
    </div>
    <UiTabs v-model="typeFormat" default-value="text" class="mb-6 flex justify-center flex-col">
      <UiTabsList class="inline-flex h-12 bg-transparent gap-4 mb-4">
        <UiTabsTrigger
          value="text"
          class="px-6 py-2 rounded-xl text-sm md:text-base font-bold transition-all duration-300 text-muted-foreground hover:text-foreground ring-2 data-[state=inactive]:ring-border/15 data-[state=inactive]:hover:ring-primary/50 data-[state=active]:text-foreground data-[state=active]:ring-primary"
        >
          <div class="flex items-center gap-2">
            <CustomIcon icon-name="lucide:align-left" ui="secondary" size="s" />
            <span>Текст</span>
          </div>
        </UiTabsTrigger>
        <UiTabsTrigger
          value="file"
          class="px-6 py-2 rounded-xl text-sm md:text-base font-bold transition-all duration-300 text-muted-foreground hover:text-foreground ring-2 data-[state=inactive]:ring-border/15 data-[state=inactive]:hover:ring-primary/50 data-[state=active]:text-foreground data-[state=active]:ring-primary"
        >
          <div class="flex items-center gap-2">
            <CustomIcon icon-name="lucide:file-up" ui="secondary" size="s" />
            <span>Файл</span>
          </div>
        </UiTabsTrigger>
      </UiTabsList>
      <UiTabsContent value="text" class="mt-0 outline-none">
        <TextAreaField v-model="analyzeText" @analyze="handleAnalyze" @clear="handleClear" />
      </UiTabsContent>
      <UiTabsContent value="file" class="mt-0 outline-none">
        <FileUploadField @analyze="handleAnalyzeFile" />
      </UiTabsContent>
    </UiTabs>
    <Transition name="slide-up">
      <div v-if="analyzeResult || analyzeIsLoading" ref="resultRef" class="scroll-mt-28">
        <BaseResult :result="analyzeResult" :loading="analyzeIsLoading" />
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.slide-up-enter-active {
  transition: all 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

.slide-up-enter-from {
  opacity: 0;
  transform: translateY(40px) scale(0.98);
}
</style>
