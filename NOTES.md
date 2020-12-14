### Problemy wstępne

- boost.compute jest nierozwijany od 2 lat
- brak szczegółowej dokumentacji w przystępnej formie
- większość przykładów używa C98 i OpenCL
- najczęstrze użycia :
    - wrapper do wołania surowego kodu OpenCL
    - wraper do OpenCl i używanie funkcji do transferu wektoróœ danych/typów graficznych [vec3/vec4]

### Problemy z konfiguracją

- brak instrukcji co do kompilacji pod windows 
- brak jasnych informacji o sterownikach OpenCL od producentów kart graficznych [Nvidia intel]

### Problemy na etapie implementacji 

- Błędy kompilacji dynamiczne [riuntime] w przypadku użytkowania liczb ujemnych ? [brak szczegółowych informacji o powodach wystąpienia, brak info w dokumentacji]