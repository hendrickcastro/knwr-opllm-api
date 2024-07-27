import ollama

response = ollama.chat(
    model='llama3.1',
    messages=[
        {"role": "system", "content": "Eres un asistente que ayuda a los usuarios con información precisa y detallada."},
        {"role": "user", "content": "Hola, ¿cómo estás?"},
        {"role": "assistant", "content": "Hola! Estoy aquí para ayudarte con cualquier pregunta que tengas."},
        {"role": "user", "content": "¿Puedes darme una recomendación de libros sobre inteligencia artificial?"}
    ],

		# provide a weather checking tool to the model
    tools=[{
      'type': 'function',
      'function': {
        'name': 'get_current_weather',
        'description': 'Get the current weather for a city',
        'parameters': {
          'type': 'object',
          'properties': {
            'city': {
              'type': 'string',
              'description': 'The name of the city',
            },
          }, 'required': ['city'],
        },
      },
    },
  ],
)

print(response['message']['tool_calls'])