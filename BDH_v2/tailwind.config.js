/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        void: '#0B0D14',
        gold: {
          300: '#FEF08A',
          400: '#FDE047',
          500: '#EAB308',
          600: '#CA8A04',
          700: '#A16207',
        },
        silver: {
          300: '#CBD5E1',
          400: '#94A3B8',
          500: '#64748B',
        },
      },
    },
  },
  plugins: [],
}
