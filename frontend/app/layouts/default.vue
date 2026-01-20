<script setup lang="ts"></script>

<template>
  <div class="min-h-screen flex flex-col">
    <div class="fixed inset-0 overflow-hidden pointer-events-none -z-10">
      <div
        class="absolute -top-[10%] -right-[10%] w-[60vw] h-[60vw] bg-primary/15 rounded-full blur-[120px] animate-pulse-slow"
      ></div>
      <div class="absolute -bottom-[10%] -left-[10%] w-[40vw] h-[40vw] bg-accent/10 rounded-full blur-[100px]"></div>
    </div>
    <Header />
    <div class="px-4 py-8 md:py-16 pt-24 md:pt-28">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.animate-pulse-slow {
  animation: pulse-slow 12s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes pulse-slow {
  0%,
  100% {
    opacity: 0.6;
    transform: scale(1);
  }
  50% {
    opacity: 1;
    transform: scale(1.05);
  }
}

/* Стиль для активной ссылки, который NuxtLink подставит сам */
.router-link-active {
  color: var(--primary);
}
</style>
