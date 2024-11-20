// App.tsx
import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import PlantDiseaseDetection from './plant';

const App = () => {
  return (
    <SafeAreaView style={styles.container}>
      <PlantDiseaseDetection />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

export default App;
