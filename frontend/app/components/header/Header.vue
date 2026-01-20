<script setup lang="ts">
import Btn from '~/components/buttons/Btn.vue';

const route = useRoute();

const navPaths = [
  {
    path: '/',
    name: 'Home',
    page: 'index'
  },
  {
    path: '/how-it-works',
    name: 'How it works'
  },
  {
    path: '/pricing',
    name: 'Pricing'
  }
];
</script>

<template>
  <header class="fixed inset-0 z-50 h-16">
    <div class="absolute inset-0 bg-background/80 backdrop-blur-xl border-b border-border" />
    <nav class="relative h-full mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
      <div class="h-full flex items-center justify-between">
        <NuxtLink class="flex items-center gap-2 group" to="/">
          <Icon icon-name="I" />
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
            <Btn text="Login" ui="secondary" />
          </NuxtLink>
          <NuxtLink to="/auth/register">
            <Btn text="Get Started"></Btn>
          </NuxtLink>
        </div>
      </div>
    </nav>
  </header>
</template>

<style scoped>
.router-link-active {
  color: var(--primary);
}
</style>
