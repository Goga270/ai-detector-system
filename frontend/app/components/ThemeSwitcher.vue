<script setup lang="ts">
import { useStore } from 'vuex';

const store = useStore();

const currentTheme = computed(() => store.state.theme);

function changeTheme(theme: string) {
  store.dispatch('setTheme', theme);
}
</script>

<template>
  <div class="flex items-center gap-2 rounded-lg border border-border bg-secondary/30 p-1">
    <button
      :class="`flex h-8 w-8 items-center justify-center transition-all rounded-md ${currentTheme === 'light' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'}`"
      @click="changeTheme('light')"
    >
      L
    </button>
    <button
      :class="`flex h-8 w-8 items-center justify-center transition-all rounded-md ${currentTheme === 'dark' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'}`"
      @click="changeTheme('dark')"
    >
      D
    </button>
  </div>
</template>

<style scoped>
.selected-theme {
  @apply bg-primary text-primary-foreground;
}
</style>
