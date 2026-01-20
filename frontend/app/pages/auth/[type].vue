<script setup lang="ts">
const route = useRoute();

const authType = computed(() => route.params.type);

if (authType.value !== 'register' && authType.value !== 'login') {
  throw showError({ statusCode: 404, statusMessage: 'Page Not Found' });
}
</script>

<template>{{ authType }}</template>

<style scoped></style>
