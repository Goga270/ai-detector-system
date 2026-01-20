<script setup lang="ts">
import { useStore } from 'vuex';

const store = useStore();

useHead({
  title: `%s | AI Detector`
});

onMounted(() => {
  store.dispatch('initTheme');
});
</script>

<template>
  <NuxtLayout>
    <NuxtPage></NuxtPage>
  </NuxtLayout>
</template>
