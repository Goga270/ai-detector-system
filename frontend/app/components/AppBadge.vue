<script setup lang="ts">
defineProps<{
  text: string;
}>();
</script>

<template>
  <div class="mb-4 inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-4 py-1.5">
    <Icon name="lucide:sparkles" />
    <span class="text-sm font-medium text-primary">{{ text }}</span>
  </div>
</template>

<style scoped></style>
