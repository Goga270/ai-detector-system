<script setup lang="ts">
import { ref, watch } from 'vue';

import Btn from '~/components/buttons/Btn.vue';
import CustomIcon from '~/components/icons/CustomIcon.vue';
import AppText from '~/components/text/AppText.vue';

const isMenuOpen = ref(false);

const route = useRoute();

const navPaths = [
  {
    path: '/',
    name: 'Главная',
    page: 'index'
  },
  {
    path: '/how-it-works',
    name: 'Технология'
  },
  {
    path: '/pricing',
    name: 'Тарифы'
  }
];

watch(
  () => route.fullPath,
  () => {
    isMenuOpen.value = false;
  }
);

watch(isMenuOpen, (val) => {
  if (import.meta.client) {
    document.body.style.overflow = val ? 'hidden' : '';
  }
});
</script>

<template>
  <header class="fixed inset-0 z-50 h-16">
    <div class="absolute inset-0 bg-background/80 backdrop-blur-xl border-b border-border/15" />
    <nav class="relative h-full mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
      <div class="h-full flex items-center justify-between">
        <NuxtLink class="flex items-center gap-2 group" to="/">
          <CustomIcon icon-name="lucide:sparkles" />
          <span class="text-xl font-bold text-foreground group-hover:text-primary transition-colors"
            >AIdetect
          </span></NuxtLink
        >
        <div class="hidden md:flex items-center gap-1">
          <NuxtLink
            v-for="path in navPaths"
            :key="path.path"
            class="px-4 py-2 text-sm font-medium transition-colors rounded-lg hover:bg-primary/5 text-muted-foreground hover:text-foreground"
            :to="path.path"
          >
            {{ path.name }}</NuxtLink
          >
        </div>
        <div class="hidden md:flex items-center gap-3">
          <ThemeSwitcher />
          <NuxtLink to="/auth/login">
            <Btn text="Login" ui="ghost" />
          </NuxtLink>
          <NuxtLink to="/auth/register">
            <Btn text="Get Started"></Btn>
          </NuxtLink>
        </div>
        <div class="flex items-center gap-3 md:hidden">
          <ThemeSwitcher />
          <button
            class="text-muted-foreground hover:text-foreground p-2 rounded-lg transition-colors"
            @click="isMenuOpen = !isMenuOpen"
          >
            <Icon v-show="isMenuOpen" name="lucide:x" class="w-7 h-7" />
            <Icon v-show="!isMenuOpen" name="lucide:menu" class="w-7 h-7" />
          </button>
        </div>
      </div>
      <Transition name="menu">
        <div
          v-if="isMenuOpen"
          class="absolute top-full left-0 right-0 z-40 bg-background/95 md:hidden backdrop-blur-xl border-b border-primary/10 rounded-b-lg"
        >
          <div class="flex flex-col py-4">
            <!-- Ссылки -->
            <div class="flex flex-col px-4 sm:px-6">
              <NuxtLink v-for="path in navPaths" :key="path.path" class="py-4" :to="path.path">
                <AppText type="body" color="muted" class="font-semibold hover:text-foreground transition-colors">
                  {{ path.name }}
                </AppText>
              </NuxtLink>
            </div>

            <div class="h-[2px] bg-border/10 w-full my-6" />

            <div class="px-4 sm:px-6 flex justify-between items-center">
              <NuxtLink to="/auth/login">
                <Btn text="Login" ui="ghost" size="l" />
              </NuxtLink>
              <NuxtLink to="/auth/register">
                <Btn text="Get Started" size="l" />
              </NuxtLink>
            </div>
          </div>
        </div>
      </Transition>
    </nav>
  </header>
</template>

<style scoped>
.menu-enter-active,
.menu-leave-active {
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.menu-enter-from,
.menu-leave-to {
  opacity: 0;
  transform: translateY(-20px);
}
</style>
