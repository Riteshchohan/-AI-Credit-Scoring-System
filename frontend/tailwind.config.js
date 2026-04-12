/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Snapmint-inspired brand palette (teal + lime)
        brand: {
          // teal
          50: '#e8f6f8',
          100: '#d1edf1',
          200: '#a6dbe3',
          300: '#73c3d1',
          400: '#3aa4b8',
          500: '#1c879f',
          600: '#0f6f86',
          700: '#0c596c',
          800: '#0a4a5a',
          900: '#073947',
        },
        accent: {
          // lime
          50: '#f6fde8',
          100: '#ecfbd0',
          200: '#d9f7a1',
          300: '#bff15f',
          400: '#a3e635',
          500: '#86cc1e',
          600: '#68a412',
          700: '#4f7d10',
          800: '#3f6212',
          900: '#365314',
        },
        success: {
          50: '#f0fdf4',
          500: '#22c55e',
          600: '#16a34a',
        },
        danger: {
          50: '#fef2f2',
          500: '#ef4444',
          600: '#dc2626',
        },
        warning: {
          50: '#fffbeb',
          500: '#f59e0b',
          600: '#d97706',
        }
      }
    },
  },
  plugins: [],
}