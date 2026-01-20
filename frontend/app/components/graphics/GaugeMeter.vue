<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';

interface Props {
  value: number; // Процент от 0 до 100
}

const props = defineProps<Props>();

// Состояние для анимации цифр
const animatedValue = ref(0);

// SVG параметры
const radius = 80;

const strokeWidth = 12;

const normalizedRadius = radius - strokeWidth / 2;

const circumference = normalizedRadius * Math.PI; // Длина полуокружности

// Вычисляем смещение (dashoffset) для прогресса
const strokeDashoffset = computed(() => {
  return circumference - (animatedValue.value / 100) * circumference;
});

// Логика цвета (используем твои переменные из main.css)
const color = computed(() => {
  if (props.value > 70) return 'var(--error)'; // Красный (ИИ)
  if (props.value > 40) return 'var(--warning)'; // Желтый (Подозрительно)

  return 'var(--success)'; // Зеленый (Человек)
});

// Плавная анимация числа при монтировании или изменении
const animate = () => {
  // Простая анимация: число "добегает" до нужного значения
  const duration = 1500;

  const start = animatedValue.value;

  const end = props.value;

  const startTime = performance.now();

  const update = (now: number) => {
    const elapsed = now - startTime;

    const progress = Math.min(elapsed / duration, 1);

    // Функция сглаживания (ease-out-expo)
    const ease = 1 - Math.pow(2, -10 * progress);

    animatedValue.value = Math.floor(start + (end - start) * ease);

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  };

  requestAnimationFrame(update);
};

onMounted(() => {
  animate();
});

// Если значение изменится (например, новый скан), запускаем анимацию снова
watch(
  () => props.value,
  () => {
    animate();
  }
);
</script>

<template>
  <div class="relative flex flex-col items-center select-none">
    <svg :width="radius * 2 + 20" :height="radius + 40" class="overflow-visible">
      <!-- Фон (серая дуга) -->
      <path
        :d="`M ${10 + strokeWidth / 2} ${radius + 10} A ${normalizedRadius} ${normalizedRadius} 0 0 1 ${radius * 2 + 10 - strokeWidth / 2} ${radius + 10}`"
        fill="none"
        stroke="var(--border)"
        :stroke-width="strokeWidth"
        stroke-linecap="round"
      />

      <!-- Прогресс (цветная дуга) -->
      <path
        :d="`M ${10 + strokeWidth / 2} ${radius + 10} A ${normalizedRadius} ${normalizedRadius} 0 0 1 ${radius * 2 + 10 - strokeWidth / 2} ${radius + 10}`"
        fill="none"
        :stroke="color"
        :stroke-width="strokeWidth"
        stroke-linecap="round"
        :stroke-dasharray="circumference"
        :stroke-dashoffset="strokeDashoffset"
        class="transition-all duration-300 ease-out"
        :style="{
          filter: `drop-shadow(0 0 12px ${color}44)`
        }"
      />

      <!-- Подписи -->
      <text
        :x="10"
        :y="radius + 40"
        fill="var(--muted-foreground)"
        font-size="11"
        class="font-medium uppercase tracking-wider"
      >
        Human
      </text>
      <text
        :x="radius * 2 + 10"
        :y="radius + 40"
        fill="var(--muted-foreground)"
        font-size="11"
        text-anchor="end"
        class="font-medium uppercase tracking-wider"
      >
        AI
      </text>
    </svg>

    <div class="absolute bottom-2 flex flex-col items-center">
      <div class="flex items-baseline">
        <span class="text-5xl font-black transition-colors duration-500 tabular-nums" :style="{ color: color }">
          {{ animatedValue }}
        </span>
        <span class="text-xl font-bold ml-0.5" :style="{ color: color }">%</span>
      </div>
      <span class="text-[10px] uppercase tracking-[0.2em] font-bold text-muted-foreground/60 mt-1">
        AI Probability
      </span>
    </div>
  </div>
</template>

<style scoped>
span {
  text-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
}
</style>
