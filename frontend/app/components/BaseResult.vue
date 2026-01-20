<script setup lang="ts">
import GaugeMeter from '~/components/graphics/GaugeMeter.vue';
import type { DetectionResult } from '~/store/modules/detector';

defineProps<{
  result: DetectionResult;
  loading: boolean;
}>();
</script>

<template>
  <div class="mt-8 rounded-2xl border border-border bg-card p-6 shadow-xl shadow-primary/5 md:p-8">
    <h2 class="mb-6 text-center text-xl font-semibold text-foreground">Analysis Results</h2>
    <template v-if="result">
      <GaugeMeter :value="result.interpretation.confidence_score * 100" />
    </template>
  </div>
</template>

<style scoped></style>
