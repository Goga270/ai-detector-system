<script setup lang="ts">
import { ref } from 'vue';

import { MailIcon } from 'lucide-vue-next';

definePageMeta({
  layout: 'auth'
});
const route = useRoute();

const authType = computed(() => route.params.type);

if (authType.value !== 'register' && authType.value !== 'login') {
  throw showError({ statusCode: 404, statusMessage: 'Page Not Found' });
}

const msg = ref('');
</script>

<template>
  <div>
    {{ msg }}
    <UiInputGroup>
      <UiInputGroupInput v-model="msg" type="email" placeholder="Enter your email" />
      <UiInputGroupAddon>
        <MailIcon />
      </UiInputGroupAddon>
    </UiInputGroup>
  </div>
</template>

<style scoped></style>
