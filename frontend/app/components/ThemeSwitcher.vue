<script setup lang="ts">
import { useStore } from 'vuex';

import Btn from '~/components/buttons/Btn.vue';

const store = useStore();

const currentTheme = computed(() => store.state.theme);

function changeTheme(theme: string) {
  store.dispatch('setTheme', theme);
}
</script>

<template>
  <div class="flex items-center gap-2 rounded-lg outline outline-1 outline-border/15 bg-secondary/30 p-1">
    <Btn :ui="currentTheme === 'light' ? 'primary' : 'ghost'" size="s" is-icon @click="changeTheme('light')">
      <Icon name="lucide:sun-medium" />
    </Btn>
    <Btn :ui="currentTheme === 'light' ? 'ghost' : 'primary'" size="s" is-icon @click="changeTheme('dark')">
      <Icon name="lucide:moon-star" />
    </Btn>
  </div>
</template>

<style scoped>
.selected-theme {
  @apply bg-primary text-primary-foreground;
}
</style>
